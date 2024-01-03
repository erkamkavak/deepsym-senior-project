from .base_trainer import BaseTrainer

class AutoEncoderTrainer(BaseTrainer): 
    # override base class methods
    def pass_batch_from_model(self, batch): 
        current_state, _, _ = batch
        input = current_state.cuda()
        output = self.model(input)
        loss = self.model.loss_function(input, output)
        return loss

    def save_batch_logs(self, batch, save_path, curr_iter, val=False): 
        current_state, _, _ = batch
        input = current_state.cuda()
        output = self.model(input)

        if type(output) == tuple:
            self.model.save_other_outputs(output, f"{self.save_path}/logs/test/", f"output_{curr_iter}")
            output = output[0]
        self.save_input_output_ground_truth(input, output, input, f"output_{curr_iter}.png", val)