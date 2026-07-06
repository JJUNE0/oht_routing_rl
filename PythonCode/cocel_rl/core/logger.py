from cocel_rl.utils.utils import create_onnx_policy
import os
import yaml
import torch

from datetime import datetime

class OffPolicyLogger:
    def __init__(self, config):
        self.config = config
        self.checkpoint_no = 0
        if config['save_model']:
            # 날짜-시간 기반 폴더
            run_name = datetime.now().strftime('%Y%m%d-%H%M%S')
            self.save_dir = os.path.join(config['save_dir'], run_name, '')
            os.makedirs(self.save_dir, exist_ok=True)
        else:
            self.save_dir = None

    def _log(self, checkpoint_dir,
             policy, critic, policy_optimizer_state_dict, critic_optimizer_state_dict, encoder_optimizer_state_dict,
             meta_data, log_alpha, alpha_optimizer_state_dict, buffer):
        try:
            os.mkdir(checkpoint_dir)
        except:
            pass
        with open(checkpoint_dir + '/metadata.yaml', 'w') as meta_f:
            yaml.safe_dump(meta_data, meta_f)
        # with open(checkpoint_dir + '/config.yaml', 'w') as config_f:
        #     yaml.safe_dump(dict(self.config), config_f)

        torch.save({'policy_state_dict': policy.state_dict(),
                    'critic_state_dict': critic.state_dict(),
                    'policy_optimizer_state_dict': policy_optimizer_state_dict,
                    'critic_optimizer_state_dict': critic_optimizer_state_dict,
                    'encoder_optimizer_state_dict': encoder_optimizer_state_dict,
                    'log_alpha': log_alpha, 'alpha_optimizer_state_dict': alpha_optimizer_state_dict,
                    'buffer': buffer}, checkpoint_dir + '/checkpoint.pt')

        torch.save(policy.state_dict(), checkpoint_dir + '/policy.pt')
        
        if self.config['algorithm']['use_onnx']:
            with torch.no_grad():
                
                device = next(policy.parameters()).device
                dummy_input = torch.randn((policy.obs_dim,), device=device)
                policy.eval()
                
                onnx_policy = create_onnx_policy(policy, algorithm=meta_data['algorithm'])
                torch.onnx.export(
                    onnx_policy,
                    dummy_input,
                    checkpoint_dir + '/onnx_policy.onnx',
                    input_names=["state"],
                    output_names=["action"]
                )
                
                policy.train()
        return

    def log(self, policy, critic, policy_optimizer_state_dict, critic_optimizer_state_dict,
            encoder_optimizer_state_dict, meta_data, log_alpha, alpha_optimizer_state_dict, buffer,
            best_update_flag=False, cur_update_flag=False):
        assert self.save_dir is not None

        if best_update_flag or cur_update_flag:
            if cur_update_flag:
                checkpoint_dir = self.save_dir + 'current_checkpoint'
                self._log(checkpoint_dir, policy, critic, policy_optimizer_state_dict, critic_optimizer_state_dict,
                          encoder_optimizer_state_dict, meta_data, log_alpha, alpha_optimizer_state_dict, buffer)

            if best_update_flag:
                checkpoint_dir = self.save_dir + 'best_checkpoint'
                self._log(checkpoint_dir, policy, critic, policy_optimizer_state_dict, critic_optimizer_state_dict,
                          encoder_optimizer_state_dict, meta_data, log_alpha, alpha_optimizer_state_dict, buffer)

        else:
            checkpoint_dir = self.save_dir + 'checkpoint_' + str(int(self.checkpoint_no + 1))
            self._log(checkpoint_dir, policy, critic, policy_optimizer_state_dict, critic_optimizer_state_dict,
                      encoder_optimizer_state_dict, meta_data, log_alpha, alpha_optimizer_state_dict, buffer)
            self.checkpoint_no += 1







