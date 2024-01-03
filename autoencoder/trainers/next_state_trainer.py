from .base_trainer import BaseTrainer
import cv2

class NextStateTrainer(BaseTrainer): 
    # override base class methods
    def pass_batch_from_model(self, batch): 
        current_state, action, next_state = batch
        input = current_state.cuda()
        action = action.cuda()
        next_state = next_state.cuda()
        output = self.model(input, action)
        loss = self.model.loss_function(next_state, output)
        return loss

    def save_batch_logs(self, batch, save_path, curr_iter, val=False): 
        current_state, action, next_state = batch
        input = current_state.cuda()
        action = action.cuda()
        next_state = next_state.cuda()
        output = self.model(input, action)

        if type(output) == tuple:
            self.model.save_other_outputs(output, f"{self.save_path}/logs/test/", f"output_{curr_iter}")
            output = output[0]
        self.save_input_output_ground_truth(input, output, next_state, f"output_{curr_iter}.png", val)