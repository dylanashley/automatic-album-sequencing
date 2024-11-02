# -*- coding: ascii -*-

from features import compute_features, get_duration
import torch
import torch.nn.functional as F


class AutoregressiveTransformerPredictor(torch.nn.Module):
    def __init__(
        self,
        num_input_features,
        d_model,
        nhead,
        num_layers,
        max_sequence_length,
        num_prototypes=4,
        dropout=0.1,
    ):
        super(AutoregressiveTransformerPredictor, self).__init__()
        self.transformer = torch.nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=False,
        )
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length
        self.input_projection = torch.nn.Linear(num_input_features, d_model)
        self.output_projection = torch.nn.Linear(d_model, max_sequence_length)

        self.encoder_pos_embeddings = torch.nn.Embedding(max_sequence_length, d_model)
        self.decoder_pos_embeddings = torch.nn.Embedding(max_sequence_length, d_model)
        self.prototype_token_embedding = torch.nn.Embedding(num_prototypes, d_model)
        self.track_embedding = torch.nn.Embedding(max_sequence_length, d_model)

    def forward(
        self,
        features,
        sequence_lengths,
        padded_permutations,
        inverted_padded_permutations,
        prototype_tokens=None,
    ):
        # features:  padded_sequence_length x batch_size x d_model
        # sequence_lengths: batch_size
        # padded_permutations:  padded_sequence_length x batch_size

        seq_len = features.shape[0]
        batch_size = features.shape[1]
        device = features.device

        features = self.input_projection(features)

        if prototype_tokens is None:
            prototype_tokens = torch.arange(
                self.prototype_token_embedding.num_embeddings
            ).to(device)
        num_prototypes = prototype_tokens.shape[0]
        prototype_embeddings = self.prototype_token_embedding(prototype_tokens)[
            None, None
        ].repeat(
            1, batch_size, 1, 1
        )  # 1 x batch_size x num_prototypes x d_model

        # mask out padding
        tgt_padding_mask = torch.arange(seq_len).to(device).unsqueeze(
            0
        ) > sequence_lengths.unsqueeze(1)
        tgt_padding_mask = tgt_padding_mask.unsqueeze(1).repeat(
            1, num_prototypes, 1
        )  # batch_size x num_prototypes x seq_len
        src_padding_mask = torch.cat(
            [
                torch.zeros(batch_size, num_prototypes, 1, dtype=torch.bool).to(device),
                tgt_padding_mask,
            ],
            dim=-1,
        )  # batch_size x num_prototypes x seq_len+1
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(device)
        causal_mask = causal_mask != 0

        steps = torch.arange(seq_len).unsqueeze(1).to(device)
        permuted_features = features[
            padded_permutations, torch.arange(batch_size).to(device), :
        ]  # seq_len x batch_size x d_model

        encoder_pos = self.encoder_pos_embeddings(steps).repeat(
            1, batch_size, 1
        )  # seq_len x batch_size x d_model
        decoder_pos = self.decoder_pos_embeddings(steps).repeat(
            1, batch_size, 1
        )  # seq_len x batch_size x d_model

        permuted_features = (
            permuted_features + encoder_pos
        )  # padded_sequence_length x batch_size x d_model
        encoder_input = permuted_features.unsqueeze(-2).repeat(
            1, 1, num_prototypes, 1
        )  # seq_len x batch_size x num_prototypes x d_model
        encoder_input = torch.cat(
            [prototype_embeddings, encoder_input], dim=0
        )  # seq_len+1 x batch_size x num_prototypes x d_model
        encoder_input = encoder_input.view(
            -1, batch_size * num_prototypes, self.d_model
        )  # (seq_len+1) x (batch_size * num_prototypes) x d_model
        decoder_input = self.track_embedding(
            inverted_padded_permutations
        )  # seq_len x batch_size x d_model
        decoder_input = torch.cat(
            [
                torch.zeros(1, batch_size, self.d_model).to(device),
                decoder_input[:-1, :, :],
            ],
            dim=0,
        )
        decoder_input = decoder_input + decoder_pos
        decoder_input = (
            decoder_input.unsqueeze(2)
            .repeat(1, 1, num_prototypes, 1)
            .view(-1, batch_size * num_prototypes, self.d_model)
        )  # (seq_len) x (batch_size * num_prototypes) x d_model

        output = self.transformer(
            encoder_input,
            decoder_input,
            src_key_padding_mask=src_padding_mask.view(batch_size * num_prototypes, -1),
            memory_key_padding_mask=src_padding_mask.view(
                batch_size * num_prototypes, -1
            ),
            tgt_mask=causal_mask,
        )
        logits = self.output_projection(output)
        logits = logits.view(seq_len, batch_size, num_prototypes, -1)
        return logits

    def generate_orderings(
        self,
        features,
        sequence_lengths,
        padded_ordering,
        temperature=1.0,
        prototype_tokens=None,
    ):
        # features:  padded_sequence_length x batch_size x d_model
        # sequence_lengths: batch_size
        # padded_permutations:  padded_sequence_length x batch_size

        seq_len = features.shape[0]
        batch_size = features.shape[1]
        device = features.device

        features = self.input_projection(features)

        if prototype_tokens is None:
            prototype_tokens = torch.arange(
                self.prototype_token_embedding.num_embeddings
            ).to(device)
        num_prototypes = prototype_tokens.shape[0]
        prototype_embeddings = self.prototype_token_embedding(prototype_tokens)[
            None, None
        ].repeat(1, batch_size, 1, 1)

        # mask out padding
        tgt_padding_mask = torch.arange(seq_len).to(features.device).unsqueeze(
            0
        ) > sequence_lengths.unsqueeze(1)
        tgt_padding_mask = tgt_padding_mask.unsqueeze(1).repeat(1, num_prototypes, 1)
        src_padding_mask = torch.cat(
            [
                torch.zeros(batch_size, num_prototypes, 1, dtype=torch.bool).to(device),
                tgt_padding_mask,
            ],
            dim=-1,
        )
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(device)
        causal_mask = causal_mask != 0

        steps = torch.arange(seq_len).unsqueeze(1).to(device)
        features = features[padded_ordering, torch.arange(batch_size).to(device), :]

        encoder_pos = self.encoder_pos_embeddings(steps).repeat(1, batch_size, 1)
        decoder_pos = self.decoder_pos_embeddings(steps).repeat(1, batch_size, 1)

        features = features + encoder_pos
        encoder_input = features.unsqueeze(-2).repeat(
            1, 1, num_prototypes, 1
        )  # padded_sequence_length x batch_size x num_prototypes x d_model
        encoder_input = torch.cat([prototype_embeddings, encoder_input], dim=0)
        encoder_input = encoder_input.view(
            -1, batch_size * num_prototypes, self.d_model
        )
        decoder_input = torch.zeros(1, batch_size, self.d_model).to(device)
        decoder_input = decoder_input + decoder_pos[0:1, :, :]
        decoder_input = (
            decoder_input.unsqueeze(2)
            .repeat(1, 1, num_prototypes, 1)
            .view(-1, batch_size * num_prototypes, self.d_model)
        )
        generated_sequences = None

        distribution_mask = (
            torch.arange(self.max_sequence_length).to(device).unsqueeze(0)
            >= sequence_lengths.unsqueeze(1)
        ).float()
        distribution_mask = -distribution_mask * 1e9
        distribution_mask = (
            distribution_mask.unsqueeze(1)
            .repeat(1, num_prototypes, 1)
            .view(batch_size * num_prototypes, -1)
        )
        sequence_log_probs = 0.0

        for i in range(seq_len):
            output = self.transformer(
                encoder_input,
                decoder_input,
                src_key_padding_mask=src_padding_mask.view(
                    batch_size * num_prototypes, -1
                ),
                memory_key_padding_mask=src_padding_mask.view(
                    batch_size * num_prototypes, -1
                ),
            )
            logits = (
                self.output_projection(output[-1]) + distribution_mask.unsqueeze(0)
            ).squeeze(0)
            probs = F.softmax(logits / temperature, dim=1)
            next_token = torch.multinomial(probs, 1).squeeze(1)
            new_log_prob = F.log_softmax(logits, dim=1)[
                torch.arange(batch_size * num_prototypes).to(device), next_token
            ]
            sequence_log_probs += (
                new_log_prob
                * (i < sequence_lengths.unsqueeze(1).repeat(1, num_prototypes).view(-1))
                .to(device)
                .float()
            )

            if generated_sequences is None:
                generated_sequences = next_token.unsqueeze(0)
            else:
                generated_sequences = torch.cat(
                    [generated_sequences, next_token.unsqueeze(0)], dim=0
                )
            if i < seq_len - 1:
                distribution_mask[
                    torch.arange(batch_size * num_prototypes).to(device), next_token
                ] = -1e9
                decoder_input = self.track_embedding(generated_sequences)
                decoder_input = torch.cat(
                    [
                        torch.zeros(1, batch_size * num_prototypes, self.d_model).to(
                            device
                        ),
                        decoder_input,
                    ],
                    dim=0,
                )
                decoder_input = decoder_input + decoder_pos[0 : i + 2, :, :].unsqueeze(
                    2
                ).repeat(1, 1, num_prototypes, 1).view(
                    -1, batch_size * num_prototypes, self.d_model
                )

        sequence_mask = torch.arange(seq_len).to(device).unsqueeze(
            1
        ) < sequence_lengths[:, None].repeat(1, num_prototypes).view(1, -1)
        generated_sequences = generated_sequences * sequence_mask

        generated_sequences = generated_sequences.view(
            seq_len, batch_size, num_prototypes
        ).permute(1, 2, 0)
        sequence_log_probs = sequence_log_probs.view(batch_size, num_prototypes)

        return generated_sequences, sequence_log_probs


class MLPFeatureEncoder(torch.nn.Module):
    def __init__(self, num_in_features, hidden_units, num_out_features):
        super(MLPFeatureEncoder, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(num_in_features, hidden_units),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_units, num_out_features),
        )

    def forward(self, x):
        return self.layers(x)


AUDIO_FEATURES_MEAN = torch.FloatTensor(
    [
        5.2791e-01,
        2.2449e-01,
        3.8183e-01,
        3.6993e-01,
        2.9982e-01,
        3.8187e-01,
        3.1002e-01,
        3.7320e-01,
        2.0208e-01,
        2.6804e-01,
        3.9923e-01,
        4.3168e-01,
        6.2267e-01,
        5.9279e-01,
        6.1113e-01,
        5.8892e-01,
        6.0950e-01,
        5.9625e-01,
        5.9040e-01,
        6.1322e-01,
        5.9029e-01,
        6.0826e-01,
        5.8856e-01,
        5.9840e-01,
        2.5661e-01,
        2.5127e-01,
        2.4965e-01,
        2.5162e-01,
        2.5456e-01,
        2.4456e-01,
        2.4378e-01,
        2.5206e-01,
        2.5097e-01,
        2.4318e-01,
        2.3381e-01,
        2.3240e-01,
        2.4800e-01,
        2.4516e-01,
        2.4000e-01,
        2.4532e-01,
        2.4597e-01,
        2.3562e-01,
        2.3633e-01,
        2.4271e-01,
        2.4510e-01,
        2.3234e-01,
        2.2480e-01,
        2.2213e-01,
        8.4064e-03,
        8.1638e-03,
        7.3796e-03,
        8.6046e-03,
        7.7356e-03,
        7.2500e-03,
        6.9234e-03,
        7.1527e-03,
        7.2182e-03,
        5.8434e-03,
        5.9789e-03,
        5.4530e-03,
        2.5467e-01,
        1.6795e-01,
        2.8478e-01,
        1.7187e-01,
        2.3761e-01,
        2.5986e-01,
        2.2495e-01,
        2.8042e-01,
        1.7172e-01,
        3.1986e-01,
        3.0718e-01,
        3.5050e-01,
        1.2598e-01,
        1.1959e-01,
        1.2556e-01,
        1.1771e-01,
        1.2499e-01,
        1.2066e-01,
        1.1934e-01,
        1.2601e-01,
        1.1908e-01,
        1.2691e-01,
        1.1909e-01,
        1.2264e-01,
        -8.4326e-02,
        -2.8620e-01,
        -2.1652e-01,
        -2.0500e-01,
        -2.3942e-01,
        -2.2943e-01,
        -1.6260e-01,
        -1.6243e-01,
        -2.7317e-01,
        -2.4140e-01,
        1.2904e-01,
        -1.9747e-01,
        9.9976e-01,
        9.9967e-01,
        9.9968e-01,
        9.9959e-01,
        9.9957e-01,
        9.9950e-01,
        9.9956e-01,
        9.9962e-01,
        9.9952e-01,
        9.9941e-01,
        9.9929e-01,
        9.9934e-01,
        5.1320e-01,
        4.9779e-01,
        5.0352e-01,
        4.9787e-01,
        5.0958e-01,
        4.9195e-01,
        4.8912e-01,
        5.0678e-01,
        4.9703e-01,
        4.9453e-01,
        4.7444e-01,
        4.7420e-01,
        4.9036e-01,
        4.7883e-01,
        4.7642e-01,
        4.7946e-01,
        4.8671e-01,
        4.6736e-01,
        4.6632e-01,
        4.8050e-01,
        4.7865e-01,
        4.6511e-01,
        4.4870e-01,
        4.4502e-01,
        2.8735e-02,
        2.8518e-02,
        2.7429e-02,
        2.8872e-02,
        2.6975e-02,
        2.6558e-02,
        2.5836e-02,
        2.5778e-02,
        2.6021e-02,
        2.3761e-02,
        2.4284e-02,
        2.3370e-02,
        2.8777e-01,
        2.9203e-01,
        3.4148e-01,
        2.9513e-01,
        2.9885e-01,
        3.5336e-01,
        3.4541e-01,
        3.2180e-01,
        2.9474e-01,
        3.7517e-01,
        3.9954e-01,
        4.3075e-01,
        2.5269e-01,
        2.3767e-01,
        2.5309e-01,
        2.3596e-01,
        2.5154e-01,
        2.4339e-01,
        2.4089e-01,
        2.5480e-01,
        2.3830e-01,
        2.5464e-01,
        2.3792e-01,
        2.4294e-01,
        -3.5454e-01,
        -2.9166e-01,
        -4.0009e-01,
        -3.1653e-01,
        -4.2214e-01,
        -2.2881e-01,
        -2.3561e-01,
        -2.8511e-01,
        -3.2785e-01,
        -3.8974e-01,
        -2.6676e-01,
        -4.1448e-01,
        9.9990e-01,
        9.9984e-01,
        9.9992e-01,
        9.9984e-01,
        9.9991e-01,
        9.9984e-01,
        9.9972e-01,
        9.9984e-01,
        9.9974e-01,
        9.9986e-01,
        9.9976e-01,
        9.9985e-01,
        4.5933e-01,
        4.4650e-01,
        4.6346e-01,
        4.5029e-01,
        4.6557e-01,
        4.4601e-01,
        4.3699e-01,
        4.5209e-01,
        4.4442e-01,
        4.6092e-01,
        4.4246e-01,
        4.5707e-01,
        4.1605e-01,
        4.0992e-01,
        4.2225e-01,
        4.1502e-01,
        4.2522e-01,
        4.0211e-01,
        3.9731e-01,
        4.0884e-01,
        4.0870e-01,
        4.1782e-01,
        4.0692e-01,
        4.1799e-01,
        4.5159e-03,
        4.5525e-03,
        4.5987e-03,
        4.6496e-03,
        4.5296e-03,
        4.3994e-03,
        4.3606e-03,
        4.3770e-03,
        4.3728e-03,
        4.4594e-03,
        4.4800e-03,
        4.5301e-03,
        4.6570e-01,
        4.5374e-01,
        4.2390e-01,
        4.3251e-01,
        4.1180e-01,
        5.1473e-01,
        4.9554e-01,
        4.6922e-01,
        4.4990e-01,
        4.4085e-01,
        4.5154e-01,
        4.2726e-01,
        2.8285e-01,
        2.6803e-01,
        2.8224e-01,
        2.6832e-01,
        2.8301e-01,
        2.7633e-01,
        2.6666e-01,
        2.7855e-01,
        2.6624e-01,
        2.8341e-01,
        2.6505e-01,
        2.7613e-01,
        4.2059e00,
        3.1306e00,
        9.1649e-01,
        7.2178e-01,
        3.5335e-01,
        4.1843e-01,
        2.8124e-01,
        3.2054e-01,
        2.6186e-01,
        3.2422e-01,
        3.1941e-01,
        3.5792e-01,
        3.8596e-01,
        4.0863e-01,
        4.1353e-01,
        4.4958e-01,
        4.5617e-01,
        4.6465e-01,
        4.9059e-01,
        5.1693e-01,
        -2.7090e01,
        2.3410e02,
        7.9375e01,
        9.6896e01,
        6.0232e01,
        6.0672e01,
        4.0107e01,
        4.5127e01,
        3.6209e01,
        3.7697e01,
        3.3004e01,
        3.5122e01,
        3.1656e01,
        3.2926e01,
        3.0887e01,
        3.1397e01,
        2.9667e01,
        3.0237e01,
        2.9511e01,
        2.9373e01,
        -2.0517e02,
        1.4657e02,
        -1.1853e01,
        2.8651e01,
        2.7555e00,
        1.3501e01,
        -2.5490e00,
        6.0590e00,
        -1.5567e00,
        2.1654e00,
        -1.8447e00,
        1.2135e00,
        -1.7650e00,
        3.3898e-01,
        -1.5383e00,
        -6.4117e-01,
        -1.8229e00,
        -8.1233e-01,
        -1.2605e00,
        -1.2488e00,
        -1.9236e02,
        1.5126e02,
        -1.3223e01,
        2.9452e01,
        2.5508e00,
        1.4005e01,
        -2.5152e00,
        6.3141e00,
        -1.5124e00,
        2.3801e00,
        -1.7860e00,
        1.2831e00,
        -1.7384e00,
        3.7994e-01,
        -1.5237e00,
        -6.3577e-01,
        -1.8225e00,
        -8.5448e-01,
        -1.3013e00,
        -1.2991e00,
        -5.2302e02,
        -1.2262e01,
        -9.5093e01,
        -4.4006e01,
        -5.6190e01,
        -3.7804e01,
        -4.6381e01,
        -3.5484e01,
        -3.9124e01,
        -3.5664e01,
        -3.6706e01,
        -3.2910e01,
        -3.4555e01,
        -3.2079e01,
        -3.2914e01,
        -3.1591e01,
        -3.1908e01,
        -2.9975e01,
        -2.9720e01,
        -2.9428e01,
        -1.2741e00,
        -1.0631e00,
        3.1980e-01,
        -2.0116e-01,
        3.9615e-02,
        -1.8376e-01,
        -9.4662e-03,
        -1.1825e-01,
        -9.2078e-03,
        -1.1064e-01,
        -2.0627e-02,
        -3.4435e-02,
        1.0169e-04,
        -1.9576e-02,
        8.7354e-03,
        1.3924e-02,
        2.4191e-02,
        5.0116e-02,
        6.0832e-02,
        6.8201e-02,
        8.3756e01,
        3.8178e01,
        2.6853e01,
        2.0042e01,
        1.6328e01,
        1.3346e01,
        1.1591e01,
        1.0624e01,
        9.7867e00,
        9.4821e00,
        8.8402e00,
        8.5959e00,
        8.2409e00,
        8.0542e00,
        7.8667e00,
        7.7183e00,
        7.5172e00,
        7.3436e00,
        7.1812e00,
        7.1067e00,
        1.8247e00,
        1.3046e01,
        4.4389e00,
        4.3175e00,
        1.4680e-02,
        4.2195e-01,
        2.1509e00,
        6.4805e00,
        3.5438e03,
        1.4179e03,
        1.3739e03,
        3.0926e02,
        1.0041e00,
        4.5953e02,
        2.3912e01,
        5.6422e03,
        1.1829e03,
        1.0708e03,
        1.5060e02,
        2.4764e00,
        6.0147e02,
        1.9216e00,
        6.8641e-01,
        6.7932e-01,
        6.2187e-01,
        8.9454e-01,
        1.6253e00,
        1.9309e00,
        5.9824e01,
        4.0551e01,
        4.3054e01,
        3.6389e01,
        3.7035e01,
        3.6902e01,
        4.8326e01,
        2.0219e01,
        1.5567e01,
        1.7814e01,
        1.7835e01,
        1.9025e01,
        1.9212e01,
        3.1688e01,
        1.9641e01,
        1.5153e01,
        1.7400e01,
        1.7479e01,
        1.8631e01,
        1.8765e01,
        3.2120e01,
        4.4796e00,
        3.0970e00,
        4.5340e00,
        5.8367e00,
        7.1566e00,
        7.8890e00,
        1.0651e01,
        8.3496e-01,
        5.8754e-01,
        5.7329e-01,
        5.2930e-01,
        5.8200e-01,
        7.2092e-01,
        -4.3871e-01,
        5.4368e00,
        4.4359e00,
        4.5296e00,
        3.8700e00,
        3.7923e00,
        3.6116e00,
        6.1406e00,
        1.7333e01,
        9.2499e03,
        2.3785e03,
        2.1591e03,
        1.6810e02,
        1.9081e00,
        1.2911e03,
        1.2932e00,
        1.6692e00,
        6.3867e-01,
        6.5748e-01,
        5.7501e-01,
        6.0553e-01,
        1.1656e-01,
        1.3492e-01,
        3.1049e-01,
        3.2006e-01,
        7.7826e-02,
        7.5679e-02,
        6.0139e-04,
        4.0781e-03,
        -8.6603e-04,
        1.2193e-02,
        2.0899e-03,
        1.7502e-03,
        7.9670e-04,
        3.9617e-03,
        -1.0464e-03,
        1.2486e-02,
        1.9542e-03,
        2.1062e-03,
        -1.2095e-01,
        -1.1600e-01,
        -3.0805e-01,
        -3.0022e-01,
        -7.1578e-02,
        -7.6487e-02,
        -6.6826e-02,
        6.6652e-02,
        2.6555e-02,
        -1.6834e-02,
        2.8301e-02,
        -9.1941e-02,
        3.0639e-02,
        3.1630e-02,
        9.9123e-02,
        9.8908e-02,
        2.1559e-02,
        2.1843e-02,
        2.8947e01,
        3.8300e-01,
        5.2321e-02,
        4.3172e-02,
        2.3379e-03,
        3.2175e00,
        4.0169e-02,
    ]
)

AUDIO_FEATURES_STD = torch.FloatTensor(
    [
        4.3470e01,
        1.3722e01,
        1.2674e01,
        2.6272e01,
        1.7290e01,
        1.6440e01,
        2.1260e01,
        1.8601e01,
        7.2909e00,
        6.6638e00,
        1.7876e01,
        2.0623e01,
        8.5248e-02,
        8.2377e-02,
        8.7710e-02,
        8.5706e-02,
        8.7838e-02,
        8.9384e-02,
        8.7886e-02,
        8.7952e-02,
        8.5874e-02,
        9.2088e-02,
        9.0287e-02,
        9.0939e-02,
        8.1673e-02,
        7.3583e-02,
        7.9699e-02,
        7.1317e-02,
        7.8855e-02,
        7.3073e-02,
        7.0956e-02,
        7.8042e-02,
        7.1633e-02,
        7.8771e-02,
        7.2639e-02,
        7.5694e-02,
        9.5525e-02,
        8.6439e-02,
        9.4248e-02,
        8.3434e-02,
        9.2845e-02,
        8.6125e-02,
        8.3340e-02,
        9.1592e-02,
        8.3952e-02,
        9.3199e-02,
        8.5293e-02,
        8.9319e-02,
        2.9890e-02,
        2.8350e-02,
        2.6796e-02,
        2.7922e-02,
        2.6442e-02,
        2.4992e-02,
        2.4750e-02,
        2.6395e-02,
        2.5324e-02,
        2.2289e-02,
        2.2100e-02,
        2.1700e-02,
        8.3979e-01,
        6.9277e-01,
        7.5152e-01,
        7.6390e-01,
        7.3331e-01,
        7.4170e-01,
        7.1543e-01,
        7.4275e-01,
        6.6417e-01,
        6.9591e-01,
        7.4144e-01,
        7.6757e-01,
        3.8831e-02,
        3.5083e-02,
        3.9053e-02,
        3.5189e-02,
        3.7830e-02,
        3.7220e-02,
        3.5460e-02,
        3.8813e-02,
        3.5079e-02,
        3.9680e-02,
        3.5762e-02,
        3.6519e-02,
        2.2676e01,
        5.8539e00,
        8.7503e00,
        1.1773e01,
        1.5691e01,
        9.7710e00,
        2.9706e01,
        2.3677e01,
        7.2182e00,
        8.8681e00,
        4.5714e01,
        6.8860e00,
        8.5960e-03,
        9.3013e-03,
        9.3798e-03,
        1.0319e-02,
        1.0649e-02,
        1.1252e-02,
        1.0278e-02,
        1.0031e-02,
        1.1339e-02,
        1.2426e-02,
        1.3400e-02,
        1.3309e-02,
        1.3836e-01,
        1.2414e-01,
        1.3487e-01,
        1.2319e-01,
        1.3562e-01,
        1.2600e-01,
        1.2232e-01,
        1.3114e-01,
        1.2010e-01,
        1.3145e-01,
        1.2076e-01,
        1.2400e-01,
        1.8032e-01,
        1.5432e-01,
        1.7532e-01,
        1.5311e-01,
        1.7487e-01,
        1.5856e-01,
        1.5254e-01,
        1.7101e-01,
        1.4962e-01,
        1.7102e-01,
        1.5021e-01,
        1.5459e-01,
        3.3199e-02,
        3.1599e-02,
        2.9989e-02,
        3.0742e-02,
        2.9203e-02,
        2.8439e-02,
        2.7867e-02,
        2.8886e-02,
        2.7956e-02,
        2.5860e-02,
        2.6145e-02,
        2.5956e-02,
        8.1555e-01,
        6.0038e-01,
        7.1351e-01,
        6.3760e-01,
        6.9914e-01,
        6.3145e-01,
        6.7235e-01,
        7.6279e-01,
        6.0393e-01,
        6.7889e-01,
        8.0700e-01,
        6.0858e-01,
        4.7283e-02,
        4.2342e-02,
        4.7017e-02,
        4.2345e-02,
        4.5271e-02,
        4.4825e-02,
        4.2186e-02,
        4.6718e-02,
        4.2119e-02,
        4.8185e-02,
        4.3846e-02,
        4.5048e-02,
        4.6618e00,
        5.6929e00,
        5.4360e00,
        4.6459e00,
        4.2683e00,
        7.5120e00,
        6.0264e00,
        1.5376e01,
        4.1081e00,
        6.2236e00,
        6.0550e00,
        3.9103e00,
        5.4995e-03,
        5.9713e-03,
        4.4395e-03,
        6.3835e-03,
        5.2336e-03,
        6.4595e-03,
        8.3001e-03,
        6.9639e-03,
        9.0048e-03,
        6.8281e-03,
        8.6979e-03,
        5.4029e-03,
        1.2832e-01,
        1.2242e-01,
        1.2607e-01,
        1.2406e-01,
        1.2798e-01,
        1.2699e-01,
        1.1973e-01,
        1.1985e-01,
        1.1788e-01,
        1.2178e-01,
        1.1819e-01,
        1.2167e-01,
        1.7109e-01,
        1.5910e-01,
        1.7035e-01,
        1.6190e-01,
        1.7204e-01,
        1.6469e-01,
        1.5406e-01,
        1.5867e-01,
        1.5098e-01,
        1.6288e-01,
        1.5368e-01,
        1.6028e-01,
        1.2525e-02,
        1.2132e-02,
        1.2734e-02,
        1.2263e-02,
        1.1954e-02,
        1.1529e-02,
        1.1285e-02,
        1.1870e-02,
        1.1411e-02,
        1.1916e-02,
        1.1976e-02,
        1.2129e-02,
        6.4485e-01,
        6.1846e-01,
        6.4703e-01,
        6.2575e-01,
        6.4689e-01,
        6.4932e-01,
        6.1623e-01,
        6.6281e-01,
        5.7424e-01,
        6.3228e-01,
        6.2194e-01,
        6.0122e-01,
        4.3245e-02,
        4.1351e-02,
        4.3142e-02,
        4.1143e-02,
        4.2074e-02,
        4.2256e-02,
        4.1795e-02,
        4.3581e-02,
        4.0658e-02,
        4.3514e-02,
        4.1618e-02,
        4.1495e-02,
        1.3125e01,
        5.5300e00,
        2.7629e00,
        1.7980e00,
        1.3088e00,
        1.2657e00,
        1.1238e00,
        1.0899e00,
        1.0151e00,
        1.3024e00,
        1.5569e00,
        1.3422e00,
        1.4027e00,
        1.2914e00,
        1.2773e00,
        1.6425e00,
        2.0586e00,
        1.8168e00,
        1.5949e00,
        1.6628e00,
        8.3053e01,
        2.6200e01,
        2.9343e01,
        2.5758e01,
        1.8048e01,
        1.6590e01,
        1.3644e01,
        1.2914e01,
        1.1627e01,
        1.1502e01,
        1.0662e01,
        1.0711e01,
        1.0422e01,
        1.0469e01,
        1.0505e01,
        1.0671e01,
        1.0399e01,
        1.0447e01,
        1.0207e01,
        1.0620e01,
        9.6303e01,
        3.3776e01,
        3.0836e01,
        1.8606e01,
        1.3506e01,
        1.1606e01,
        9.4844e00,
        8.7508e00,
        6.9363e00,
        7.5652e00,
        5.8364e00,
        6.1931e00,
        5.3129e00,
        5.3705e00,
        4.6808e00,
        4.8657e00,
        4.3670e00,
        4.5381e00,
        3.9776e00,
        4.1981e00,
        1.0265e02,
        3.6223e01,
        3.2727e01,
        1.9538e01,
        1.4073e01,
        1.2040e01,
        9.7084e00,
        8.9589e00,
        7.0430e00,
        7.6829e00,
        5.8855e00,
        6.2697e00,
        5.3393e00,
        5.4076e00,
        4.6805e00,
        4.8743e00,
        4.3587e00,
        4.5226e00,
        3.9522e00,
        4.1571e00,
        5.1423e01,
        2.8996e01,
        3.0078e01,
        2.4324e01,
        1.9909e01,
        1.6279e01,
        1.4592e01,
        1.2923e01,
        1.1250e01,
        1.1887e01,
        1.0374e01,
        1.0164e01,
        9.9717e00,
        9.6319e00,
        9.0728e00,
        8.9268e00,
        8.4867e00,
        7.9513e00,
        7.7094e00,
        7.4812e00,
        1.2157e00,
        9.6218e-01,
        7.7362e-01,
        6.3593e-01,
        4.9832e-01,
        4.6734e-01,
        4.0382e-01,
        3.8800e-01,
        3.3849e-01,
        3.6094e-01,
        3.2611e-01,
        3.3748e-01,
        3.2137e-01,
        3.2528e-01,
        3.1436e-01,
        3.2172e-01,
        3.2271e-01,
        3.2150e-01,
        3.2517e-01,
        3.3812e-01,
        2.4312e01,
        1.0781e01,
        8.2675e00,
        6.0008e00,
        4.2173e00,
        3.5706e00,
        2.8949e00,
        2.6504e00,
        2.2962e00,
        2.3165e00,
        2.0586e00,
        2.0618e00,
        2.0208e00,
        2.0158e00,
        1.9743e00,
        1.9458e00,
        1.8643e00,
        1.7903e00,
        1.7283e00,
        1.7159e00,
        1.6164e01,
        4.7436e00,
        2.4681e00,
        2.6576e00,
        2.1792e-01,
        1.0453e00,
        1.0544e00,
        2.2958e01,
        3.7592e02,
        4.4980e02,
        4.9695e02,
        3.1643e02,
        1.7212e00,
        1.7479e02,
        1.0428e02,
        1.0983e03,
        5.1575e02,
        5.3173e02,
        1.9652e02,
        2.8117e00,
        2.8973e02,
        1.3948e00,
        8.6099e-01,
        1.0354e00,
        1.4098e00,
        1.7786e00,
        2.9030e00,
        5.6201e00,
        8.2205e00,
        4.9927e00,
        5.2972e00,
        5.3389e00,
        6.2565e00,
        7.5513e00,
        6.9606e00,
        3.4767e00,
        2.6058e00,
        2.8583e00,
        2.7857e00,
        2.9274e00,
        3.3248e00,
        6.7623e00,
        3.5912e00,
        2.7359e00,
        3.0092e00,
        2.9260e00,
        3.0786e00,
        3.4685e00,
        7.5545e00,
        1.9138e00,
        1.2503e00,
        1.7989e00,
        2.2284e00,
        2.7213e00,
        3.0857e00,
        3.9691e00,
        3.4312e-01,
        2.8189e-01,
        3.0710e-01,
        4.1949e-01,
        5.2279e-01,
        7.0270e-01,
        1.1153e00,
        8.6089e-01,
        8.2195e-01,
        8.8904e-01,
        1.0214e00,
        1.1841e00,
        1.5233e00,
        2.2323e00,
        1.0491e02,
        1.0075e03,
        1.1211e03,
        1.2555e03,
        3.2405e02,
        2.7857e00,
        5.5955e02,
        3.7390e00,
        4.9528e00,
        1.9708e00,
        2.0886e00,
        2.0241e00,
        2.0860e00,
        4.2591e-02,
        5.8639e-02,
        1.0051e-01,
        9.9365e-02,
        2.5102e-02,
        2.4983e-02,
        1.3298e-02,
        1.5338e-02,
        6.2493e-02,
        6.5644e-02,
        1.0379e-02,
        1.0956e-02,
        1.3116e-02,
        1.4939e-02,
        6.6074e-02,
        6.8765e-02,
        1.0543e-02,
        1.1061e-02,
        4.5927e-02,
        4.5771e-02,
        9.9120e-02,
        1.0299e-01,
        2.3158e-02,
        2.4883e-02,
        5.5687e-01,
        6.5749e-01,
        5.3498e-01,
        5.4129e-01,
        4.1941e-01,
        4.3130e-01,
        8.0535e-03,
        8.8540e-03,
        3.7039e-02,
        3.6595e-02,
        5.9891e-03,
        6.0295e-03,
        7.9627e01,
        1.8288e-01,
        3.0984e-02,
        2.9936e-02,
        5.0204e-03,
        2.9834e00,
        2.6461e-02,
    ]
)


def get_features(filename: str) -> float:
    """Returns a narrative arc value for an audio file."""
    a = torch.from_numpy(compute_features(filename).to_numpy()).float()
    a = (a - AUDIO_FEATURES_MEAN) / AUDIO_FEATURES_STD
    durations_mean = 257.6875
    durations_std = 191.8393
    duration = get_duration(filename)
    duration = (duration - durations_mean) / durations_std
    duration = torch.FloatTensor([duration])
    features = torch.cat([a, duration.repeat(7)]).unsqueeze(0)
    return features


def get_orderings(
    features: torch.Tensor,
    feature_encoder: torch.nn.Module,
    ordering_predictor: torch.nn.Module,
    num_orderings_to_generate=10,
    num_choices=1,
) -> torch.Tensor:
    num_narrative_features = 1
    num_features = 525
    dev = features.device
    track_features = features
    track_numbers = torch.arange(track_features.shape[0]).to(dev)
    seq_lengths = torch.tensor([track_features.shape[0]]).to(dev)
    batch_size = seq_lengths.shape[0]

    padded_features = track_features

    narrative_features = feature_encoder(padded_features.view(-1, num_features))
    narrative_features = narrative_features.view(
        -1, batch_size, num_narrative_features
    )  # padded_sequence_length x batch_size x 1

    feature_mask = torch.arange(narrative_features.shape[0]).unsqueeze(1).to(
        dev
    ) < seq_lengths.unsqueeze(0)
    feature_mask = feature_mask.unsqueeze(2).float()
    valid_features = narrative_features * feature_mask

    permutations = [torch.randperm(l) for l in seq_lengths]
    inverted_permutations = [torch.argsort(p) for p in permutations]
    padded_permutations = torch.nn.utils.rnn.pad_sequence(permutations).to(dev)
    padded_inverted_permutations = torch.nn.utils.rnn.pad_sequence(
        inverted_permutations
    ).to(dev)

    all_orderings = []
    all_log_probs = []
    for _ in range(num_orderings_to_generate):
        orderings, log_probs = ordering_predictor.generate_orderings(
            valid_features, seq_lengths, padded_permutations, temperature=1.0
        )
        all_orderings.append(orderings)
        all_log_probs.append(log_probs)
    orderings = torch.cat(all_orderings, dim=1)
    log_probs = torch.cat(
        all_log_probs, dim=1
    )  # batch_size x num_orderings_to_generate

    best_orderings = []
    log_probs_of_best_orderings = []
    for c in range(num_choices):
        best_idx = torch.argmax(log_probs, dim=1)
        best_log_probs = log_probs[torch.arange(batch_size), best_idx]
        best_ordering = orderings[torch.arange(batch_size), best_idx]
        best_orderings.append(best_ordering)
        log_probs_of_best_orderings.append(best_log_probs)
        log_probs[log_probs == best_log_probs.unsqueeze(1)] = -float("inf")

    orderings = torch.stack(best_orderings, dim=1).squeeze(0)
    log_probs_of_best_orderings = torch.stack(
        log_probs_of_best_orderings, dim=1
    ).squeeze(0)
    return (
        orderings.detach().cpu().numpy(),
        log_probs_of_best_orderings.detach().cpu().numpy(),
        narrative_features.squeeze(1, 2).detach().cpu().numpy(),
    )
