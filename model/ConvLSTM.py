import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, dilation):
        """
        input_dim: Anzahl der Kanäle, die reinkommen (deine 34).
        hidden_dim: Wie groß das Gedächtnis pro Pixel sein soll (z.B. 64).
        kernel_size: Die Lupe (3x3).
        dilation: Lücken in der Lupe für größeres Sichtfeld.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = (
            input_dim  # number of channels (variables) of input (per pixel)
        )
        self.hidden_dim = hidden_dim  # memory-like: to each pixel it has saved hidden_dim information (it is usually double the input_dim
        self.kernel_size = kernel_size  # Always looks at a pixel and the 8 neighbouring pixles (3 x 3) kernel (usually 3 or 5)
        self.dilation = dilation  # 1

        self.conv = nn.Conv2d(
            in_channels=self.input_dim
            + self.hidden_dim,  # receives input channels of current timestep PLUS the memory of previous timestep
            out_channels=4
            * self.hidden_dim,  # 4 output channels: for the 4 gates (input, forget, cell and output)
            kernel_size=self.kernel_size,
            padding="same",
            dilation=self.dilation,
            bias=True,  # Adds learnable bias to the output
        )

    def forward(self, input_tensor, cur_state):  # is calles once for each timestep
        """
        input_tensor Shape: (Batch, Channels, Height, Width) -> (B, 34, 256, 256)
        cur_state: Tuple of (h_cur, c_cur), both have shape:    (B, 64, 256, 256)
        """

        # Getting the current state
        h_cur, c_cur = (
            cur_state  # h_cur (hidden state): Is the idea of the current kNDVI. Dimension: (Batch, hidden_dim (64), patch_size (256), patch_size (256)
        )
        # c_cur (Cell State): Is Longterm memory. Internal memory, which stays stable over long periods

        # Combine input and previous state/ memory
        combined = torch.cat(
            [input_tensor, h_cur], dim=1
        )  # Glue the input_dims (34 channels) with the 64 channels (hidden_dims) from  the previous memory -> 98 channels

        # Convolution
        combined_conv = self.conv(
            combined
        )  # Uses learned filters (weights) to calculate from the 98 (combined) input channels the 256 (4 * hidden_dim) output channels
        # Input: (Batch, 98, 256, 256)
        # Output: (Batch, 256, 256, 256)

        # As we multiplied by 4, we now can split into: Forget, Input, current (new info) and output gates
        cc_i, cc_f, cc_o, cc_g = torch.split(
            combined_conv, self.hidden_dim, dim=1
        )  # cc_f (Forget Gate): Decides what of c_cur can be forgotten and is no longer relevant
        # cc_i (Input Gate): Decides, which new information from this timestep is relevant
        # cc_g (New Info): Calculates the new values
        # cc_o (Output Gate): Decides, what we pass as h_next (visible result)

        # Activation function
        i = torch.sigmoid(cc_i)  # Sigmoid transforms all number between 0 and 1
        f = torch.sigmoid(cc_f)  # 0: Let nothing pass
        o = torch.sigmoid(cc_o)  # 1: Let everything pass
        g = torch.tanh(cc_g)  # Tanh: Transforms values to -1 and 1

        # Updating the long term memory
        c_next = f * c_cur + i * g  # Calculating the cell state:
        # f * c_cur (forgetting): Takes old memory (c_cur) and multiplies with Forget Gate (f). If f is 0 at one spot this information gets lost
        # i * g (remembering): Model takes new information (g) and multiplies it with info gate. Only adding what is important
        # Plus (+): Combines old with new knowledge
        # Calculating the visible result
        h_next = o * torch.tanh(
            c_next
        )  # Hidden state (h) is what the model shows to outside and what will be used for the final prediction
        # Takes the updated memory and transforms it with tang and the filters it with output_gate (o)

        return h_next, c_next

    def init_hidden(self, batch_size, height, width):
        return (
            torch.zeros(
                batch_size,
                self.hidden_dim,
                height,
                width,
                device=self.conv.weight.device,
            ),
            torch.zeros(
                batch_size,
                self.hidden_dim,
                height,
                width,
                device=self.conv.weight.device,
            ),
        )


class ConvLSTM(nn.Module):
    def __init__(
        self,
        input_dim,
        decoder_input_dim,
        output_dim,
        hidden_dims,
        kernel_size,
        dilation,
        num_layers,
        baseline="last_frame",
    ):
        """
        input_dim: Channels in context tensor (z.B. 34)
        decoder_input_dim: Channel in future (for prediction) (z.B. 25: 1 Pred + Climate + Statics)
        """
        super(ConvLSTM, self).__init__()

        # --- Safety ---
        if not isinstance(hidden_dims, list) or len(hidden_dims) != num_layers:
            raise ValueError(f"hidden_dims muss eine Liste der Länge num_layers ({num_layers}) sein.")

        self._check_kernel_size_consistency(kernel_size)
        self.input_dim = input_dim
        self.decoder_input_dim = decoder_input_dim
        self.hidden_dims = hidden_dims  # Expects a list - 1 for each layer
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.baseline = baseline

        # ENCODER LAYERS (für die Vergangenheit)
        self.encoder_cells = nn.ModuleList()
        for i in range(self.num_layers):
            cur_in = self.input_dim if i == 0 else self.hidden_dims[i - 1]
            self.encoder_cells.append(
                ConvLSTMCell(cur_in, self.hidden_dims[i], kernel_size, dilation)
            )

        # DECODER LAYERS (für die Zukunft - angepasste Eingangsgröße!)
        self.decoder_cells = nn.ModuleList()
        for i in range(self.num_layers):
            cur_in = self.decoder_input_dim if i == 0 else self.hidden_dims[i - 1]
            self.decoder_cells.append(
                ConvLSTMCell(cur_in, self.hidden_dims[i], kernel_size, dilation)
            )

        # Verwandelt abstrakte "Gedächtnis" (zb. 64 Kanäle in hidden state) in tatsächlichen kNDVI (1 Kanal) um
        self.predict_layer = nn.Conv2d(
            in_channels=hidden_dims[-1],  # Die channels from previous hidden state
            out_channels=output_dim,  # 1 channel for kNDVI
            kernel_size=1,  # 1x1 bedeutet: schaue nur auf den Pixel selbst
        )

    def forward(self, input_tensor, prediction_count, non_pred_feat, baseline_sample):
        """
        input_tensor: (B, T_ctx, C, H, W)          -> Past observation with T_ctx observations, and C channels/variables
        prediction_count: int                      -> Number of prediction timesteps in the future
        non_pred_feat: (B, T_fut, C_npf, H, W)     -> Future weather / statics
        baseline_sample:                            -> Baseline (last_frame)
        """

        # --- ABSICHERUNG ---
        # input_tensor: (B, T_ctx, C, H, W)
        if input_tensor.dim() != 5:
            raise ValueError(f"Input Tensor muss 5D sein, got {input_tensor.dim()}D")
        if non_pred_feat.size(1) != prediction_count:
            raise ValueError(f"Nicht genug Wetterdaten! Prediction braucht {prediction_count} Schritte, "
                            f"aber non_pred_feat hat nur {non_pred_feat.size(1)}.")

        # 1.  Extract dimensions
        # b: Batch, t_ctx: number of timesteps, c: number of input vars, height/width: patch size (256 or 128)
        b, t_ctx, _, height, width = input_tensor.size()  # (B, T_ctx, C, H, W)
        baseline = baseline_sample.to(input_tensor.device)

        # -- 2. Calculate baseline --
        if self.baseline == "last_frame":  # UDPATE (s. Kladny)
            # Last kNDVI frame from context
            # Input Shape: (B, T_ctx, C, H, W) -> Output Shape: (B, 1 (kNDVI), H, W)
            baseline = baseline_sample
        elif self.baseline == "mean_cube":
            # Mean kNDVI over all T_ctx past timesteps
            # Input Shape: (B, T_ctx, C, H, W) -> Output Shape: (B, 1 (kNDVI), H, W)
            baseline = torch.mean(input_tensor[:, :, 0:1, :, :], dim=1)
        else:
            # Start with 0
            # Shape: (B, 1 (kNDVI), H, W)
            baseline = torch.zeros((b, 1, height, width), device=input_tensor.device)

        if baseline.size(-1) != width or baseline.size(1) != 1:
            raise ValueError("Baseline Shape Mismatch! Check Channels and Spatial Dim.")

        # -- 3. INITIALIZE STATES--
        # For each layer we need a Hidden State (h) and Cell State (c)
        hs = []  # List of tensors: [(B, 64, H, W), (B, 64, H, W), ...]  # 64 = hidden_dim
        cs = []  # List of tensors: [(B, 64, H, W), (B, 64, H, W), ...]
        for i in range(self.num_layers):
            hx, cx = self.encoder_cells[i].init_hidden(b, height, width)
            hs.append(hx)
            cs.append(cx)

        # --- 4. ENCODER PHASE (PROCESS CONTEXT) ---
        # We go step by step through past (context) timesteps and update memory (h and c)
        for t in range(t_ctx):
            # Shape: (B, C, H, W) -> because timestep t was selected
            input_t = input_tensor[:, t, :, :, :]

            # Now this timesteps gets passed through layers
            for i in range(self.num_layers):
                # Cell calculates new h and c based on input and old h/c
                # hs[i] Shape: (B, 64, H, W)  64: hidden_dim
                hs[i], cs[i] = self.encoder_cells[i](input_t, (hs[i], cs[i]))

                # Result of layer i becomes input for layer i+1
                input_t = hs[i]

        # --- Prepare decoder ---
        # Create storage for results
        # Shape: (B, T_fut, 1 (kNDVI), H, W)
        preds = torch.zeros(
            (b, prediction_count, 1, height, width), device=input_tensor.device
        )
        pred_deltas = torch.zeros(
            (b, prediction_count, 1, height, width), device=input_tensor.device
        )
        baselines = torch.zeros(
            (b, prediction_count, 1, height, width), device=input_tensor.device
        )

        #  Start decoder at t=0:
        for t in range(prediction_count):
            # What do we provide model as input
            if t == 0:
                # Am ersten Tag nehmen wir die Baseline + das Wetter von heute (t=0)
                # Wir müssen sicherstellen, dass baseline und non_pred_feat zusammenpassen
                guided_input = torch.cat([baseline, non_pred_feat[:, 0, :, :, :]], dim=1)
                curr_baseline = baseline
            else:
                # In future timesteps we use prediction of day before as baseline + new weather
                guided_input = torch.cat([preds[:, t - 1, :, :, :], non_pred_feat[:, t, :, :, :]], dim=1)
                curr_baseline = preds[:, t - 1, :, :, :]

            # --- SAFTEY CHECK ---
            if guided_input.size(1) != self.decoder_input_dim:
                raise ValueError(f"Guided Input Channel Mismatch at timestep {t}! Expected {self.decoder_input_dim}, got {guided_input.size(1)}")
            
            # Run through decoder layers
            input_t = guided_input
            for i in range(self.num_layers):
                hs[i], cs[i] = self.decoder_cells[i](input_t, (hs[i], cs[i]))
                input_t = hs[i]

            # Calculate delta and add to baseline to get final prediction
            delta = self.predict_layer(hs[-1])  # Get delta prediction from last layer
            pred_deltas[:, t, :, :, :] = delta  # Store delta prediction
            baselines[:, t, :, :, :] = curr_baseline  # Store baseline for this timestep
            preds[:, t, :, :, :] = curr_baseline + delta  # Final prediction

        return preds, pred_deltas, baselines
    
        # # --- 4. FIRST PREDICTION (day 1 in the future) ---
        # baselines[:, 0, :, :, :] = baseline
        # # Das oberste h (letzter Layer) nach dem Encoder-Durchlauf ist unser Delta
        # # delta Shape: (B, 1 (kNDVI), H, W)
        # delta = self.predict_layer(hs[-1])
        # pred_deltas[:, 0, :, :, :] = delta  #  erste vorhersage nutzt keine future features

        # # Final Prediction = Baseline + Delta
        # # Shape: (B, 1 (kNDVI), H, W)
        # preds[:, 0, :, :, :] = (
        #     pred_deltas[:, 0, :, :, :] + baselines[:, 0, :, :, :]
        # )  # set kNDVI value (delta + baseline) for timestep 0 in preds

        # # -- 6. DECODER PHASE (Guided Future Prediction) --
        # if prediction_count > 1:

        #     for t in range(
        #         1, prediction_count
        #     ):  # first prediction is already done see above
        #         # GUIDING: Combine last prediction with new climate vars
        #         # preds[:, t-1, :, :, :]: (B, 1 (kNDVI), H, W)
        #         # non_pred_feat[:, t, :, :, :]]: (B, C_npf, H, W)
        #         # combined_input: (B, C_npf + 1, H, W)
        #         guided_input = torch.cat(
        #             [preds[:, t - 1, :, :, :], non_pred_feat[:, t, :, :, :]], dim=1
        #         )

        #         # Put through all layers
        #         input_t = guided_input
        #         for i in range(self.num_layers):
        #             hs[i], cs[i] = self.decoder_cells[i](input_t, (hs[i], cs[i]))
        #             input_t = hs[i]

        #         # Update Baseline (Use last prediction as new baseline)
        #         baselines[:, t, :, :, :] = preds[:, t - 1, :, :, :]
        #         current_delta = self.predict_layer(hs[-1])
        #         pred_deltas[:, t] = current_delta
        #         preds[:, t, :, :, :] = (
        #             baselines[:, t, :, :, :] + pred_deltas[:, t, :, :, :]
        #         )

        # # Return complete tensor with predictions for the timesteps
        # # Shape: (Batch, prediction_count, 1 (kNDVI), 256, 256)
        # return preds, pred_deltas, baselines

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (
            isinstance(kernel_size, tuple)
            or isinstance(kernel_size, int)
            or (
                isinstance(kernel_size, list)
                and all([isinstance(elem, tuple) for elem in kernel_size])
            )
        ):
            raise ValueError("`kernel_size` must be tuple or list of tuples")

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
