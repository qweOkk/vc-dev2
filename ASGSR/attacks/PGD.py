import torch
from attacks.Attack import Attack
from attacks import loss_functions
from attacks import score_functions
from attacks import decision_functions


class PGD(Attack):
    r"""
    PGD is a class to implement PGD attack.
    Examples::
        >>> attack = torchattacks.PGD(model, eps=0.008, alpha=0.0004, steps=20, random_start=True, until_success=False)
    """

    def __init__(self, model, **kwargs):
        super().__init__('PGD', model)
        self.eps = kwargs['eps']
        self.alpha = kwargs['alpha']
        self.steps = kwargs['steps']
        self.random_start = kwargs['random_start']
        self.until_success = kwargs['until_success']

        loss_function_name = kwargs['loss_function']['name']
        loss_function_config = kwargs['loss_function'][loss_function_name]
        self.loss_function = getattr(loss_functions, loss_function_name)(loss_function_config, threshold=self.model.threshold)

        score_function_name = kwargs['score_function']['name']
        score_function_config = kwargs['score_function'][score_function_name]
        self.score_function = getattr(score_functions, score_function_name)(score_function_config, threshold=self.model.threshold)

        decision_function_name = kwargs['decision_function']['name']
        decision_function_config = kwargs['decision_function'][decision_function_name]
        self.decision_function = getattr(decision_functions, decision_function_name)(decision_function_config, threshold=self.model.threshold)

        self.init_info()

    def forward(self, enroll_waveforms, eval_waveforms, labels):
        enroll_waveforms = enroll_waveforms.clone().detach().to(self.device)
        eval_waveforms = eval_waveforms.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        adv_eval_waveforms = eval_waveforms.clone().detach()

        if self.random_start:
            adv_eval_waveforms = adv_eval_waveforms + torch.empty_like(adv_eval_waveforms).uniform_(-self.eps, self.eps)
            adv_eval_waveforms = torch.clamp(adv_eval_waveforms, min=-1, max=1).detach()

        step = 0

        enroll_embeddings = self.model(enroll_waveforms)
        while True:
            adv_eval_waveforms.requires_grad = True

            adv_eval_embeddings = self.model(adv_eval_waveforms)

            similarity_scores = self.score_function(enroll_embeddings, adv_eval_embeddings)
            cost = self.loss_function(similarity_scores, labels)

            grad = torch.autograd.grad(cost, adv_eval_waveforms, retain_graph=False, create_graph=False)[0]

            adv_eval_waveforms = adv_eval_waveforms.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_eval_waveforms - eval_waveforms, min=-self.eps, max=self.eps)
            adv_eval_waveforms = torch.clamp(eval_waveforms + delta, min=-1, max=1).detach()

            step += 1

            # attack until success
            # Note: if a batch has both successful and unsuccessful attacks, the successful ones will be attacked too, until they are all successful
            # Todo: Attack only those samples in the batch that have not yet been successfully attacked.
            if self.until_success:
                adv_eval_embeddings = self.model(adv_eval_waveforms)
                similarity_scores = self.score_function(enroll_embeddings, adv_eval_embeddings)
                decisions = self.decision_function(enroll_embeddings, adv_eval_embeddings)
                is_successes = torch.logical_xor(decisions.bool(), labels.bool())
                if torch.all(is_successes):
                    self.update_info(step)
                    return adv_eval_waveforms, is_successes, similarity_scores

            if step == self.steps and not self.until_success:
                break

        adv_eval_embeddings = self.model(adv_eval_waveforms)
        similarity_scores = self.score_function(enroll_embeddings, adv_eval_embeddings)
        decisions = self.decision_function(enroll_embeddings, adv_eval_embeddings)
        is_successes = torch.logical_xor(decisions.bool(), labels.bool())
        self.update_info(step)
        return adv_eval_waveforms, is_successes, similarity_scores

    def init_info(self):
        self.max_step = 10000000
        self.min_step = 0
        self.sum_step = 0
        self.count = 0  # number of samples

    def update_info(self, step):
        self.sum_step += step
        self.count += 1
        if step < self.min_step:
            self.min_step = step
        if step > self.max_step:
            self.max_step = step

    def display_info(self):
        return 'Avg. steps: {:.2f}, Min steps: {}, Max steps: {}'.format(self.sum_step / self.count, self.min_step, self.max_step)
