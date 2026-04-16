import torch
import torch.nn as nn


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


def _extend_for_multilayer(param, num_layers):
    if not isinstance(param, list):
        param = [param] * num_layers
    return param


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, dilation, layer_norm_flag):
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
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.layer_norm_flag = layer_norm_flag

        self.conv = nn.Conv2d(
            in_channels=self.input_dim
            + self.hidden_dim,  # receives input channels of current timestep PLUS the memory of previous timestep
            out_channels=4
            * self.hidden_dim,  # 4 output channels: for the 4 gates (input, forget, cell and output)
            kernel_size=self.kernel_size,
            padding="same",
            padding_mode="reflect",
            dilation=self.dilation,
            bias=True,  # Adds learnable bias to the output
        )

        if self.layer_norm_flag:
            self.layer_norm = nn.InstanceNorm2d(
                self.input_dim + self.hidden_dim, affine=True
            )

        self._init_weights()

    def forward(self, input_tensor, cur_state):
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

        if self.layer_norm_flag:
            combined = self.layer_norm(combined)

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
        # Takes the updated memory and transforms it with tanh and the filters it with output_gate (o)

        return h_next, c_next

    def _init_weights(self):
        # nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        # nn.init.xavier_normal_(self.conv.weight)
        nn.init.orthogonal_(self.conv.weight)
        if self.conv.bias is not None:
            h = self.hidden_dim
            self.conv.bias.data.fill_(0)
            self.conv.bias.data[h : 2 * h].fill_(
                1.0
            )  # set bias of forget gate to 1.0 -> at beginning model keeps information

    def init_hidden(self, batch_size, height, width):
        # Erzeuge neutrale Nullen direkt auf dem richtigen Device
        h_start = torch.zeros(
            batch_size, self.hidden_dim, height, width, device=self.conv.weight.device
        )
        c_start = torch.zeros(
            batch_size, self.hidden_dim, height, width, device=self.conv.weight.device
        )
        return h_start, c_start


class SGEDConvLSTM(nn.Module):
    def __init__(
        self,
        input_dim,
        decoder_input_dim,
        output_dim,
        hidden_dims,
        kernel_size,
        dilation,
        num_layers,
        cfg,
        baseline="last_frame",
        dropout_prob=0.0,
        layer_norm_flag=False,
    ):
        """
        input_dim: Channels in context tensor (z.B. 34)
        decoder_input_dim: Channel in future (for prediction) (z.B. 25: 1 Pred + Climate + Statics)
        """
        super(SGEDConvLSTM, self).__init__()

        # --- Safety ---
        if not isinstance(hidden_dims, list) or len(hidden_dims) != num_layers:
            raise ValueError(
                f"hidden_dims muss eine Liste der Länge num_layers ({num_layers}) sein."
            )

        _check_kernel_size_consistency(kernel_size)
        self.cfg = cfg
        self.input_dim = input_dim
        self.decoder_input_dim = decoder_input_dim
        self.hidden_dims = hidden_dims  # Expects a list - 1 for each layer
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.num_layers = num_layers
        self.baseline = baseline
        self.layer_norm_flag = layer_norm_flag

        # self.batch_norms = nn.ModuleList([
        #     nn.BatchNorm2d(hidden_dims[i]) for i in range(self.num_layers)
        # ])

        self.dropout_prob = dropout_prob

        # Dropout layers for all but the last layer (no information drop right before predition)
        self.dropout_layers = nn.ModuleList(
            [nn.Dropout2d(p=self.dropout_prob) for _ in range(self.num_layers - 1)]
        )

        # ENCODER LAYERS (für die Vergangenheit)
        self.encoder_cells = nn.ModuleList()
        for i in range(self.num_layers):
            cur_in = self.input_dim if i == 0 else self.hidden_dims[i - 1]
            self.encoder_cells.append(
                ConvLSTMCell(
                    cur_in,
                    self.hidden_dims[i],
                    self.kernel_size,
                    self.dilation,
                    self.layer_norm_flag,
                )
            )

        # DECODER LAYERS (für die Zukunft - angepasste Eingangsgröße!)
        self.decoder_cells = nn.ModuleList()
        for i in range(self.num_layers):
            cur_in = self.decoder_input_dim if i == 0 else self.hidden_dims[i - 1]
            self.decoder_cells.append(
                ConvLSTMCell(
                    cur_in,
                    self.hidden_dims[i],
                    self.kernel_size,
                    self.dilation,
                    self.layer_norm_flag,
                )
            )

        # Verwandelt abstrakte "Gedächtnis" (zb. 64 Kanäle in hidden state) in tatsächlichen kNDVI (1 Kanal) um
        self.predict_layer = nn.Conv2d(
            in_channels=self.hidden_dims[-1],  # Die channels from previous hidden state
            out_channels=output_dim,  # 1 channel for kNDVI
            kernel_size=1,
        )

        # # Maybe delete
        nn.init.normal_(self.predict_layer.weight, mean=0.0, std=0.05)
        nn.init.constant_(self.predict_layer.bias, 0.0)
        # nn.init.constant_(self.predict_layer.weight, 0.0)
        # nn.init.constant_(self.predict_layer.bias, 0.0)

    def forward(self, input_tensor, prediction_count, non_pred_feat, baseline_sample):
        """
        input_tensor: (B, T_ctx, C, H, W)          -> Past observation with T_ctx observations, and C channels/variables
        prediction_count: int                      -> Number of prediction timesteps in the future
        non_pred_feat: (B, T_fut, C_npf, H, W)     -> Future weather / statics
        baseline_sample:                            -> Baseline (last_frame)
        """

        # --- SAFETY ---
        # input_tensor: (B, T_ctx, C, H, W)
        if input_tensor.dim() != 5:
            raise ValueError(f"Input Tensor muss 5D sein, got {input_tensor.dim()}D")
        if non_pred_feat.size(1) != prediction_count:
            raise ValueError(
                f"Nicht genug Wetterdaten! Prediction braucht {prediction_count} Schritte, "
                f"aber non_pred_feat hat nur {non_pred_feat.size(1)}."
            )

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
            # Mean kNDVI over all T_ctx context timesteps
            # Input Shape: (B, T_ctx, C, H, W) -> Output Shape: (B, 1 (kNDVI), H, W)
            baseline = torch.mean(input_tensor[:, :, 0:1, :, :], dim=1)
        else:
            # Start with 0
            # Shape: (B, 1 (kNDVI), H, W)
            baseline = torch.zeros((b, 1, height, width), device=input_tensor.device)

        if baseline.size(-1) != width or baseline.size(1) != 1:
            raise ValueError(
                "Baseline Shape Mismatch! Check Channels and Spatial Dimensions."
            )

        # -- 3. INITIALIZE STATES--
        # For each layer we need a Hidden State (h) and Cell State (c)
        hs = (
            []
        )  # List of tensors: [(B, 64, H, W), (B, 64, H, W), ...]  # 64 = hidden_dim
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

                # input_t = self.batch_norms[i](hs[i])

                # Result of layer i becomes input for layer i+1
                if i < self.num_layers - 1:
                    input_t = self.dropout_layers[i](hs[i])
                    # input_t = self.dropout_layers[i](input_t) #  batch_norm
                else:
                    input_t = hs[i]
                    # input_t = self.batch_norms[i](hs[i]) # batch_norm

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

            guided_input = non_pred_feat[:, t, :, :, :].clone()
            # What do we provide model as input
            if t == 0:
                # Am ersten Tag nehmen wir die Baseline + das Wetter von heute (t=0)
                # Wir müssen sicherstellen, dass baseline und non_pred_feat zusammenpassen
                # Set baseline (last valid kNDVI) to kNDVI channel
                guided_input[:, 0:1, :, :] = baseline
                curr_baseline = baseline
            else:
                # In future timesteps we use prediction of day before as baseline + new weather
                ## DAS IST EIN TEST: (probiere mal aus!!)
                # prev_pred = preds[:, t - 1, :, :, :].detach()
                prev_pred = preds[:, t - 1, :, :, :]

                # Das hier ausprobierne:
                # Wenn du Exposure Bias testen willst, kannst du hier ground_truth[:, t-1] einspeisen!
                # prev_pred = ground_truth[:, t-1]

                # Preidction overwrites kNDVI channel (kNDVI is at same first position as in the encoder input tensor
                guided_input[:, 0:1, :, :] = prev_pred
                curr_baseline = prev_pred

                # ### HERE MAYBE:
                # curr_baseline = baseline

            # --- SAFTEY CHECK ---
            if guided_input.size(1) != self.decoder_input_dim:
                raise ValueError(
                    f"Guided Input Channel Mismatch at timestep {t}! Expected {self.decoder_input_dim}, got {guided_input.size(1)}"
                )

            # Run through decoder layers
            input_t = guided_input
            for i in range(self.num_layers):
                hs[i], cs[i] = self.decoder_cells[i](input_t, (hs[i], cs[i]))

                # input_t = self.batch_norms[i](hs[i]) # batch_norm

                if i < self.num_layers - 1:
                    # input_t = self.dropout_layers[i](input_t)
                    input_t = self.dropout_layers[i](hs[i])
                else:
                    input_t = hs[i]
                    # input_t = self.batch_norms[i](hs[i]) # batch_norm

            # Calculate delta and add to baseline to get final prediction
            delta = self.predict_layer(hs[-1])  # Get delta prediction from last layer

            #  USE TANH TO KEEP DELTA IN CHECK (BETWEEN -1 AND 1) -> HELPS WITH STABILITY
            # MAYBE EVEN ADD: * 0.1 -> to really limit the step size of the predictions and make training more stable
            # delta = torch.tanh(delta)

            pred_deltas[:, t, :, :, :] = delta  # Store delta prediction
            baselines[:, t, :, :, :] = curr_baseline  # Store baseline for this timestep
            preds[:, t, :, :, :] = curr_baseline + delta  # Final prediction

        return preds, pred_deltas, baselines


class SGConvLSTM(nn.Module):
    def __init__(
        self,
        input_dim,  # Context-Channels (34)
        output_dim,  # 1 (kNDVI)
        hidden_dims,  # Liste der hidden dims pro Layer
        kernel_size,
        dilation,
        num_layers,
        cfg,
        baseline="last_frame",
        dropout_prob=0.0,
        layer_norm_flag=False,
    ):
        super(SGConvLSTM, self).__init__()

        # self._check_kernel_size_consistency(kernel_size)
        _check_kernel_size_consistency(kernel_size)

        self.cfg = cfg
        self.input_dim = input_dim
        self.hidden_dims = _extend_for_multilayer(hidden_dims, num_layers)
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.num_layers = num_layers
        self.layer_norm_flag = layer_norm_flag
        self.baseline = baseline
        self.dropout_prob = dropout_prob

        # self.batch_norms = nn.ModuleList([
        #     nn.BatchNorm2d(hidden_dims[i]) for i in range(self.num_layers)
        # ])

        assert len(self.hidden_dims) == self.num_layers, (
            f"Mismatch: hidden_dims has {len(self.hidden_dims)} entries, "
            f"but num_layers is set to {self.num_layers}."
        )

        self.cell_list = nn.ModuleList()
        for i in range(0, self.num_layers):
            cur_in = self.input_dim if i == 0 else self.hidden_dims[i - 1]
            cur_layer_norm_flag = self.layer_norm_flag if i != 0 else False

            self.cell_list.append(
                ConvLSTMCell(
                    cur_in,
                    self.hidden_dims[i],
                    self.kernel_size,
                    self.dilation,
                    cur_layer_norm_flag,
                )
            )

        self.dropout_layers = nn.ModuleList(
            [nn.Dropout2d(p=self.dropout_prob) for _ in range(self.num_layers - 1)]
        )

        # New: Final projection layer (similar to MLP in Pellicer-Valero)
        self.predict_layer = nn.Conv2d(
            in_channels=self.hidden_dims[
                -1
            ],  # Output from last ConvLSTM layer (e.g., 64)
            out_channels=output_dim,  # Final output channels (1 for kNDVI)
            kernel_size=1,  # 1x1 pixel wise convolution to map hidden state to prediction
        )
        # Think about that:
        # Die Luxus-Variante (Ein kleines Conv-MLP wie bei Pellicer-Valero)
        # Erlaubt dem Modell, am Ende noch komplexer zu "denken"
        # last_hidden_dim = self.hidden_dims[-1] // 2
        # self.predict_layer = nn.Sequential(
        #     nn.Conv2d(self.hidden_dims[-1], last_hidden_dim, kernel_size=1), # Reduziere von 64 auf 32
        #     nn.ReLU(), # Eine moderne Aktivierungsfunktion
        #     nn.Conv2d(last_hidden_dim, output_dim, kernel_size=1)            # Reduziere von 32 auf 1
        # )

    def forward(self, input_tensor, prediction_count, non_pred_feat, baseline_sample):
        # --- INPUT ASSERTS ---
        assert (
            input_tensor.dim() == 5
        ), f"input_tensor must be 5D (B, T, C, H, W), got {input_tensor.dim()}D"
        assert (
            input_tensor.size(2) == self.input_dim
        ), f"Input tensor channels must match input_dim ({self.input_dim})"
        assert input_tensor.size(-1) == input_tensor.size(
            -2
        ), "Input tensor must have square spatial dimensions"
        assert (
            non_pred_feat.size(1) == prediction_count
        ), "Future features must match prediction count"
        assert (
            baseline_sample.size(1) == 1
        ), "Baseline must have exactly 1 channel (kNDVI)"

        b, t_ctx, _, height, width = input_tensor.size()

        # 1. Initialize Baseline & State
        baseline = baseline_sample.to(input_tensor.device)

        if self.baseline == "last_frame":
            baseline = baseline_sample
        elif self.baseline == "mean_cube":
            baseline = torch.mean(input_tensor[:, :, 0:1, :, :], dim=1)
        else:
            baseline = torch.zeros((b, 1, height, width), device=input_tensor.device)

        hs, cs = [], []
        for i in range(self.num_layers):
            hx, cx = self.cell_list[i].init_hidden(b, height, width)
            hs.append(hx)
            cs.append(cx)

        # 2. Iterate over the past
        for t in range(t_ctx):
            input_t = input_tensor[:, t]
            for i in range(self.num_layers):
                hs[i], cs[i] = self.cell_list[i](input_t, (hs[i], cs[i]))
                input_t = (
                    self.dropout_layers[i](hs[i]) if i < self.num_layers - 1 else hs[i]
                )

        # 3. Prepare prediction storage
        preds = torch.zeros(
            (b, prediction_count, 1, height, width), device=input_tensor.device
        )
        pred_deltas = torch.zeros_like(preds)
        baselines = torch.zeros_like(preds)

        # Create Padding-Tensor ONCE before the loop to avoid creating it at every prediction step
        # non_pred_feat has shapee (B, T, C_fut, H, W)
        b, _, c_fut, height, width = non_pred_feat.size()
        padding_size = self.input_dim - c_fut  # how many channels need to be padded

        if padding_size > 0:
            zeros_padding = torch.zeros(
                (b, padding_size, height, width), device=non_pred_feat.device
            )

            # Implement mask logic
            # Where is mask kNDVI in padding tensor?
            # Padding tensor contains: [S2_Rest] + [S1] + [m_kndvi] + [m_s2_rest] + [m_s1]
            # See dataset.py FINAL CONTEXT STACK line ~ 560
            c_s2_rest = (
                len(self.cfg["data"]["variables"]["s2"]) - 1
            )  # number of s2 channels without kNDVI
            c_s1 = len(self.cfg["data"]["variables"]["s1"])  # number of s1 channels

            # Index of m_kndvi is right after S2 rest and S1 channels
            idx_m_kndvi = c_s2_rest + c_s1

            # Setze m_kndvi auf 1.0 (denn unsere kNDVI Prediction ist "wolkenfrei/gütig")
            # Set m_kndvi to 1.0 (because our baseline and kNDVI prediction is "cloud-free/valid")
            # m_s2_rest and m_s1 stay at 0.0 (as they are not available)
            zeros_padding[:, idx_m_kndvi, :, :] = 1.0

        # 4. Iterate over future (Guided prediction)
        for t in range(prediction_count):

            # Get future tensor (which does not match dim of input tensor yet) for current timesteo
            guided_input = non_pred_feat[:, t, :, :, :].clone()

            # Create prediction input (baseline/prediction (of t-1) + weather)
            curr_ref = baseline if t == 0 else preds[:, t - 1]
            # DETACH to avoid gradient explosion
            # curr_ref = baseline if t == 0 else preds[:, t - 1].detach()

            # Set baseline (t=0) or kNDVI prediction of t-1 if t > 0 at index 0 (overwrites kNDVI placeholder)
            guided_input[:, 0:1, :, :] = curr_ref

            # BUILD GUIDED INPUT (Padding to match dim of input tensor)
            if padding_size > 0:
                # Add zeros at the end (so they match channel position of the original input tensor)
                # only m_kndvi ist 1.0, rest of the padding is 0.0
                input_t = torch.cat([guided_input, zeros_padding], dim=1)
            else:
                input_t = guided_input

            assert (
                input_t.size(1) == self.input_dim
            ), f"Padding failed! Expected {self.input_dim}, got {input_t.size(1)}"

            # Pass through layers (Shared cells)
            for i in range(self.num_layers):
                hs[i], cs[i] = self.cell_list[i](input_t, (hs[i], cs[i]))
                input_t = (
                    self.dropout_layers[i](hs[i]) if i < self.num_layers - 1 else hs[i]
                )

            # Delta Vorhersage
            # delta = hs[-1]
            # NEU: Wir schicken den 64-Channel Output der letzten Zelle durch die finale Conv!
            # hs[-1] hat Shape (Batch, hidden_dims[-1], H, W). Wir machen daraus (Batch, 1, H, W)
            delta = self.predict_layer(hs[-1])
            # delta = torch.tanh(delta)  # Optional

            #  Save results for timestep t
            pred_deltas[:, t] = delta
            baselines[:, t] = curr_ref
            preds[:, t] = curr_ref + delta

        return preds, pred_deltas, baselines
