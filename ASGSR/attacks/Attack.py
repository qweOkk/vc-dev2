import torch


class Attack(object):

    def __init__(self, name, model, device=None):
        '''
        :param name: name of the attack
        :param model: model to attack
        :param device: device to run the attack
        '''
        self.attack = name
        self.model = model
        self.modelname = model.__class__.__name__
        if device:
            self.device = device
        else:
            try:
                self.device = next(model.parameters()).device
            except Exception:
                self.device = None
                print('Failed to set device automatically, please try set_device() manual.')

    def forward(self, enroll_waveforms, eval_waveforms, labels):
        r"""
        It defines the computation performed at every call.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def __call__(self, enroll_waveforms, eval_waveforms, labels, *args, **kwargs):
        adv_eval_waveforms, is_successes, similarity_scores = self.forward(enroll_waveforms, eval_waveforms, labels)
        average_pertubations = torch.mean(torch.abs(adv_eval_waveforms - eval_waveforms), dim=2)
        return adv_eval_waveforms, is_successes, similarity_scores, average_pertubations
