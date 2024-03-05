import torch
import torch.nn as nn
from attacks.Attack import Attack
from attacks import loss_functions
from attacks import score_functions
from attacks import decision_functions


class FGSM(Attack):
    r"""
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 0.01)

    Shape:
        - waveforms: :math:`(N, C, L)` where `N = number of batches`, `C = number of channels, only support single channel`,`L = length`. It must have a range [-1, 1].
        - labels: :math:`(N)` where each value :math:`0` is different speaker, :math:`1` is same speaker.
        - output: :math:`(N, 1, L)`.

    Examples::
        >>> attack = torchattacks.FGSM(model, eps=0.01)
        >>> adv_eval_waveforms, is_successes = attack(enroll_waveforms, eval_waveforms, labels)

    """

    def __init__(self, model, **kwargs):
        super().__init__('FGSM', model)
        self.eps = kwargs['eps']

        loss_function_name = kwargs['loss_function']['name']
        loss_function_config = kwargs['loss_function'][loss_function_name]
        self.loss_function = getattr(loss_functions, loss_function_name)(loss_function_config)

        score_function_name = kwargs['score_function']['name']
        try:
            score_function_config = kwargs['score_function'][score_function_name]
            self.score_function = getattr(score_functions, score_function_name)(score_function_config)
        except KeyError:
            self.score_function = getattr(score_functions, score_function_name)()

        decision_function_name = kwargs['decision_function']['name']
        try:
            decision_function_config = kwargs['decision_function'][decision_function_name]
            self.decision_function = getattr(decision_functions, decision_function_name)(decision_function_config)
        except KeyError:
            self.decision_function = getattr(decision_functions, decision_function_name)()

    def forward(self, enroll_waveforms, eval_waveforms, labels):
        enroll_waveforms = enroll_waveforms.clone().detach().to(self.device)
        eval_waveforms = eval_waveforms.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        eval_waveforms.requires_grad = True

        enroll_embeddings = self.model(enroll_waveforms)
        eval_embeddings = self.model(eval_waveforms)

        similarity_scores = self.score_function(enroll_embeddings, eval_embeddings)
        cost = self.loss_function(similarity_scores, labels)

        # Update adversarial images
        grad = torch.autograd.grad(cost, eval_waveforms, retain_graph=False, create_graph=False)[0]

        adv_eval_waveforms = eval_waveforms + self.eps * grad.sign()
        adv_eval_waveforms = torch.clamp(adv_eval_waveforms, min=-1, max=1).detach()

        adv_eval_embeddings = self.model(adv_eval_waveforms)
        decisions = self.decision_function(enroll_embeddings, adv_eval_embeddings)  # 0 for different, 1 for same
        is_successes = (decisions != labels)
        return adv_eval_waveforms, is_successes, similarity_scores, {}  # 0 for not success, 1 for success, {} for no additional info
