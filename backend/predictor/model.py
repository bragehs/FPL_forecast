import torch
import torch.nn as nn

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        # x: (batch_size, seq_len=5, input_dim)
        _, (hidden, cell) = self.lstm(x)
        return hidden, cell


class LSTMDecoder(nn.Module):
    def __init__(self, decoder_input_dim, hidden_dim, output_dim=1, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(decoder_input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, future_fd, hidden, cell, output_len, targets=None, teacher_forcing_ratio=0.5):
        """
        future_fd: (batch, output_len)
        targets: (batch, output_len) or None during inference
        """
        batch_size = future_fd.size(0)
        outputs = []
        
        decoder_input = torch.zeros((batch_size, 1, 1), device=future_fd.device)

        for t in range(output_len):
            fd_step = future_fd[:, t].unsqueeze(1).unsqueeze(2)  # (batch, 1, 1)
            input_combined = torch.cat([decoder_input, fd_step], dim=2)  # (batch, 1, 2)

            output, (hidden, cell) = self.lstm(input_combined, (hidden, cell))
            pred = self.fc(output.squeeze(1))  # (batch,)
            outputs.append(pred)

            if targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = targets[:, t].unsqueeze(1).unsqueeze(2)  # (batch, 1, 1)
            else:
                decoder_input = pred.unsqueeze(1)  # (batch, 1, 1)

        return torch.stack(outputs, dim=1)  # (batch, output_len)



class Seq2SeqLSTM(nn.Module):
    def __init__(self, encoder_input_dim, decoder_input_dim, hidden_dim, output_dim=1):
        super().__init__()
        self.encoder = LSTMEncoder(encoder_input_dim, hidden_dim)
        self.decoder = LSTMDecoder(decoder_input_dim, hidden_dim, output_dim)

    def forward(self, encoder_inputs, future_fd, targets=None, teacher_forcing_ratio=0.5):
        """
        encoder_inputs: (batch, 5, encoder_input_dim)
        future_fd: (batch, 3)
        targets: (batch, 3) or None during inference
        """
        hidden, cell = self.encoder(encoder_inputs)
        outputs = self.decoder(future_fd, hidden, cell,
                               output_len=future_fd.size(1),
                               targets=targets,
                               teacher_forcing_ratio=teacher_forcing_ratio)
        return outputs.squeeze(-1)  
    
class LSTMEncoderOnly(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        _, (hidden, _) = self.lstm(x)  # hidden: (num_layers, batch, hidden_dim)
        hidden_last = hidden[-1]       # (batch, hidden_dim)
        out = self.fc(hidden_last)     # (batch, output_dim)
        return out

if __name__ == "__main__":
    #do some testing
    input_tensor = torch.randn(32, 5, 26)  # batch_size=32, seq_len=5, feature_dim=25
    future_fd_tensor = torch.randn(32, 1) # batch_size=32, output_len=3
    target_tensor = torch.randn(32, 1)  # batch_size=32, output_len=3

    
    model = Seq2SeqLSTM(encoder_input_dim=26, decoder_input_dim=2, hidden_dim=256, output_dim=1)

    training_output = model(input_tensor, future_fd_tensor, targets=target_tensor, teacher_forcing_ratio=0.5)
    inference_output = model(input_tensor, future_fd_tensor)
    print("Training output shape:", training_output.shape) 
    print("Inference output shape:", inference_output.shape)  

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in model: {num_params}")


    model = LSTMEncoderOnly(input_dim=26, hidden_dim=128, output_dim=1)

    training_output = model(input_tensor)
    inference_output = model(input_tensor)
    print("Training output shape:", training_output.shape) 
    print("Inference output shape:", inference_output.shape)  

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in model: {num_params}")