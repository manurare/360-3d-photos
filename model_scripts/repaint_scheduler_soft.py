from diffusers.schedulers.scheduling_repaint import *


class RePaintSchedulerSoft(RePaintScheduler):
    def step(
            self,
            model_output: torch.Tensor,
            timestep: int,
            sample: torch.Tensor,
            original_image: torch.Tensor,
            mask: torch.Tensor,
            generator: Optional[torch.Generator] = None,
            return_dict: bool = True,
    ):

        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`int`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            original_image (`torch.Tensor`):
                The original image to inpaint on.
            mask (`torch.Tensor`):
                The mask where a value of 0.0 indicates which part of the original image to inpaint.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_repaint.RePaintSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_repaint.RePaintSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_repaint.RePaintSchedulerOutput`] is returned,
                otherwise a tuple is returned where the first element is the sample tensor.
        """
        t = timestep
        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5

        # 3. Clip "predicted x_0"
        if self.config.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        # We choose to follow RePaint Algorithm 1 to get x_{t-1}, however we
        # substitute formula (7) in the algorithm coming from DDPM paper
        # (formula (4) Algorithm 2 - Sampling) with formula (12) from DDIM paper.
        # DDIM schedule gives the same results as DDPM with eta = 1.0
        # Noise is being reused in 7. and 8., but no impact on quality has
        # been observed.

        # 5. Add noise
        device = model_output.device
        noise = randn_tensor(model_output.shape, generator=generator, device=device, dtype=model_output.dtype)
        std_dev_t = self.eta * self._get_variance(timestep) ** 0.5

        variance = 0
        if t > 0 and self.eta > 0:
            variance = std_dev_t * noise

        # 6. compute "direction pointing to x_t" of formula (12)
        # from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2) ** 0.5 * model_output

        # 7. compute x_{t-1} of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_unknown_part_mean = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        prev_unknown_part_std = std_dev_t
        prev_unknown_part = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction + variance

        # 8. Algorithm 1 Line 5 https://arxiv.org/pdf/2201.09865.pdf
        prev_known_part_mean = (alpha_prod_t_prev ** 0.5) * original_image
        prev_known_part_std = ((1 - alpha_prod_t_prev) ** 0.5)
        prev_known_part = (alpha_prod_t_prev ** 0.5) * original_image + ((1 - alpha_prod_t_prev) ** 0.5) * noise

        # 9. Algorithm 1 Line 8 https://arxiv.org/pdf/2201.09865.pdf
        lerp_mean = mask * prev_known_part_mean + (1.0 - mask) * prev_unknown_part_mean
        lerp_std = (mask ** 2 * prev_known_part_std ** 2 + (1 - mask) ** 2 * prev_unknown_part_std ** 2) ** 0.5
        pred_prev_sample = lerp_mean + lerp_std * noise
        # pred_prev_sample = mask * prev_known_part + (1.0 - mask) * prev_unknown_part

        if not return_dict:
            return (
                pred_prev_sample,
                pred_original_sample,
            )

        return RePaintSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)