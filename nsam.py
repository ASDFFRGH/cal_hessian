import torch

class NSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, device = None,noise_type = 'FINITE', **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        noise_list = []

        defaults = dict(rho=rho, adaptive=adaptive, noise_list=noise_list, **kwargs)
        super(NSAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.noise_type = noise_type

        for group in self.param_groups:
            for i,p in enumerate(group['params']):
                group['noise_list'].append(torch.zeros(p.shape, device=device))

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        for group in self.param_groups:
            for i,p in enumerate(group['params']):
                torch.randn(p.shape,out=group['noise_list'][i])
            #print(torch.max(group['noise_list'][0]))

        noise_norm = self._noise_norm()
        for group in self.param_groups:
            scale = group["rho"] / (noise_norm + 1e-12)

            for i,p in enumerate(group["params"]):
                self.state[p]["old_p"] = p.data.clone()
                #e_w = group['noise_list'][i] * scale.to(p)
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * group['noise_list'][i] * scale.to(p)
                p.add_(e_w) 

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _noise_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        if self.noise_type=='FINITE':
            norm = torch.norm( torch.stack( [ ((torch.abs(p) if group["adaptive"] else 1.0) * noise).norm(p=2).to(shared_device) for group in self.param_groups for (p,noise) in zip(group['params'],group['noise_list']) ] ), p=2 )
            return norm
        elif self.noise_type=='GAUSS':
            return torch.scalar_tensor(1.0).to(shared_device)
        ## norm = torch.norm( torch.stack([ ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device) for group in self.param_groups for p in group["params"] if p.grad is not None]), p=2)    

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups