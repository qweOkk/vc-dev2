import torch

from attacks.Attack import Attack


class MultiAttack(Attack):
    r"""
    MultiAttack is a class to attack a model with various attacks agains same images and labels.

    Arguments:
        model (nn.Module): model to attack.
        attacks (list): list of attacks.

    Examples::
        >>> atk1 = torchattacks.PGD(model, eps=8/255, alpha=2/255, iters=40, random_start=True)
        >>> atk2 = torchattacks.PGD(model, eps=8/255, alpha=2/255, iters=40, random_start=True)
        >>> atk = torchattacks.MultiAttack([atk1, atk2])
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, attacks, **kwargs):
        super().__init__("MultiAttack", attacks[0].model)
        self.attacks = attacks
        self.until_all_success = kwargs['until_all_success']

        self.check_validity()

    def check_validity(self):
        if len(self.attacks) < 2:
            raise ValueError("More than two attacks should be given.")

        # ids = [id(attack.model) for attack in self.attacks]
        # print(ids)
        # if len(set(ids)) != 1:
        #     raise ValueError("At least one of attacks is referencing a different model.")

    def forward(self, enroll_waveforms, eval_waveforms, labels):
        r"""
        Overridden.
        """
        batch_size = enroll_waveforms.shape[0]
        enroll_waveforms = enroll_waveforms.clone().detach().to(self.device)
        adv_eval_waveforms = eval_waveforms.clone().detach().to(self.device)  # save final adversarial waveforms and intermediate adversarial waveforms
        similarity_scores = torch.zeros(batch_size, len(self.attacks)).to(self.device)
        labels = labels.clone().detach().to(self.device)
        fails_index = torch.arange(batch_size).to(self.device)

        while True:
            for _, attack in enumerate(self.attacks):
                adv_eval_waveforms[fails_index], _, _, _ = attack(enroll_waveforms[fails_index], adv_eval_waveforms[fails_index], labels[fails_index])

            # check succeeds
            is_successes = torch.zeros(batch_size, len(self.attacks)).to(self.device)
            for i, attack in enumerate(self.attacks):
                enroll_embeddings = attack.model(enroll_waveforms[fails_index])
                adv_eval_embeddings = attack.model(adv_eval_waveforms[fails_index])
                similarity_scores[:, i] = attack.score_function(enroll_embeddings, adv_eval_embeddings)
                decisions = attack.decision_function(enroll_embeddings, adv_eval_embeddings)
                is_successes[:, i] = torch.logical_xor(decisions, labels[fails_index])
            is_successes = torch.min(is_successes, dim=1)[0].bool()  # if all attacks succeed, then the adversarial waveform is successful
            fails_mask = ~is_successes
            fails_index = torch.masked_select(fails_index, fails_mask)

            if self.until_all_success:
                if len(fails_index) == 0:
                    break
            else:
                break

        return adv_eval_waveforms, is_successes, similarity_scores
