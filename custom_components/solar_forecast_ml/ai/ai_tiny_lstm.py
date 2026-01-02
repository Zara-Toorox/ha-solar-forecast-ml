# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

_LOGGER = logging.getLogger(__name__)

_np: Optional[Any] = None


def _ensure_numpy() -> Any:
    """Lazy import numpy @zara"""
    global _np
    if _np is None:
        import numpy as np
        _np = np
    return _np


class TinyLSTM:
    """Multi-Output LSTM neural network with 20 features for solar forecasting @zara

    Supports multiple output neurons for per-group predictions.
    Input: 17 base features + 3 group-specific features = 20 total
    Output: num_outputs predictions (1 per panel group, or 1 for total)
    """

    def __init__(
        self,
        input_size: int = 20,
        hidden_size: int = 32,
        sequence_length: int = 24,
        num_outputs: int = 1,
        learning_rate: float = 0.005,
        dropout: float = 0.2,
        use_attention: bool = False
    ):
        """Initialize Multi-Output LSTM with given parameters @zara

        Args:
            input_size: Number of input features (default 20 for base+group)
            hidden_size: LSTM hidden layer size
            sequence_length: Number of timesteps in input sequence
            num_outputs: Number of output predictions (1 per panel group)
            learning_rate: Adam optimizer learning rate
            dropout: Dropout rate for regularization
            use_attention: Enable scaled dot-product attention mechanism
        """
        np = _ensure_numpy()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.num_outputs = num_outputs
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.use_attention = use_attention

        limit = np.sqrt(6 / (input_size + hidden_size))
        concat_size = input_size + hidden_size

        # LSTM gate weights (unchanged)
        self.Wf = np.random.uniform(-limit, limit, (hidden_size, concat_size))
        self.Wi = np.random.uniform(-limit, limit, (hidden_size, concat_size))
        self.Wc = np.random.uniform(-limit, limit, (hidden_size, concat_size))
        self.Wo = np.random.uniform(-limit, limit, (hidden_size, concat_size))

        self.bf = np.ones((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

        # Output layer - now supports multiple outputs!
        # Shape: (num_outputs, hidden_size) instead of (1, hidden_size)
        limit_out = np.sqrt(6 / (hidden_size + num_outputs))
        self.Wy = np.random.uniform(-limit_out, limit_out, (num_outputs, hidden_size))
        self.by = np.zeros((num_outputs, 1))

        # Attention mechanism weights (only initialized if use_attention=True)
        if self.use_attention:
            # Scaled Dot-Product Attention weights
            limit_attn = np.sqrt(6 / (hidden_size + hidden_size))
            # W_query: (hidden_size, hidden_size) - transforms last hidden state to query
            self.W_query = np.random.uniform(-limit_attn, limit_attn, (hidden_size, hidden_size))
            # W_key: (hidden_size, hidden_size) - transforms all hidden states to keys
            self.W_key = np.random.uniform(-limit_attn, limit_attn, (hidden_size, hidden_size))
            # W_value: (hidden_size, hidden_size) - transforms all hidden states to values
            self.W_value = np.random.uniform(-limit_attn, limit_attn, (hidden_size, hidden_size))
            # W_attn_out: (hidden_size, hidden_size*2) - projects [h; context] back to hidden_size
            self.W_attn_out = np.random.uniform(-limit_attn, limit_attn, (hidden_size, hidden_size * 2))
            # b_attn_out: (hidden_size, 1) - bias for attention output
            self.b_attn_out = np.zeros((hidden_size, 1))

        # Adam optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8

        self.m = {
            'Wf': np.zeros_like(self.Wf), 'Wi': np.zeros_like(self.Wi),
            'Wc': np.zeros_like(self.Wc), 'Wo': np.zeros_like(self.Wo),
            'bf': np.zeros_like(self.bf), 'bi': np.zeros_like(self.bi),
            'bc': np.zeros_like(self.bc), 'bo': np.zeros_like(self.bo),
            'Wy': np.zeros_like(self.Wy), 'by': np.zeros_like(self.by)
        }

        self.v = {
            'Wf': np.zeros_like(self.Wf), 'Wi': np.zeros_like(self.Wi),
            'Wc': np.zeros_like(self.Wc), 'Wo': np.zeros_like(self.Wo),
            'bf': np.zeros_like(self.bf), 'bi': np.zeros_like(self.bi),
            'bc': np.zeros_like(self.bc), 'bo': np.zeros_like(self.bo),
            'Wy': np.zeros_like(self.Wy), 'by': np.zeros_like(self.by)
        }

        # Add attention weight optimizer states if attention is enabled
        if self.use_attention:
            self.m['W_query'] = np.zeros_like(self.W_query)
            self.m['W_key'] = np.zeros_like(self.W_key)
            self.m['W_value'] = np.zeros_like(self.W_value)
            self.m['W_attn_out'] = np.zeros_like(self.W_attn_out)
            self.m['b_attn_out'] = np.zeros_like(self.b_attn_out)

            self.v['W_query'] = np.zeros_like(self.W_query)
            self.v['W_key'] = np.zeros_like(self.W_key)
            self.v['W_value'] = np.zeros_like(self.W_value)
            self.v['W_attn_out'] = np.zeros_like(self.W_attn_out)
            self.v['b_attn_out'] = np.zeros_like(self.b_attn_out)

        _LOGGER.info(
            f"TinyLSTM initialized: input={input_size}, hidden={hidden_size}, "
            f"seq={sequence_length}, outputs={num_outputs}, attention={use_attention}"
        )

    def _sigmoid(self, x: Any) -> Any:
        """Sigmoid activation @zara"""
        np = _ensure_numpy()
        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))

    def _tanh(self, x: Any) -> Any:
        """Tanh activation @zara"""
        np = _ensure_numpy()
        x_clipped = np.clip(x, -500, 500)
        return np.tanh(x_clipped)

    def _compute_attention(
        self,
        query: Any,
        hidden_states: List[Any]
    ) -> Tuple[Any, Any, Dict[str, Any]]:
        """Scaled Dot-Product Attention @zara

        Args:
            query: Last hidden state (hidden_size, 1)
            hidden_states: List of seq_len hidden states, each (hidden_size, 1)

        Returns:
            context: Weighted context vector (hidden_size, 1)
            attn_weights: Attention weights (seq_len,) for interpretability
            cache: For backpropagation
        """
        np = _ensure_numpy()

        seq_len = len(hidden_states)

        # Stack hidden states: (hidden_size, seq_len)
        H = np.hstack(hidden_states)  # (hidden_size, seq_len)

        # Compute Query: Q = W_query @ query  -> (hidden_size, 1)
        Q = np.dot(self.W_query, query)  # (hidden_size, 1)

        # Compute Keys: K = W_key @ H  -> (hidden_size, seq_len)
        K = np.dot(self.W_key, H)  # (hidden_size, seq_len)

        # Compute Values: V = W_value @ H  -> (hidden_size, seq_len)
        V = np.dot(self.W_value, H)  # (hidden_size, seq_len)

        # Scaled dot-product attention scores: scores = Q^T @ K / sqrt(d_k)
        # Q^T: (1, hidden_size), K: (hidden_size, seq_len) -> scores: (1, seq_len)
        d_k = self.hidden_size
        scores = np.dot(Q.T, K) / np.sqrt(d_k)  # (1, seq_len)

        # Numerical stability: subtract max before exp
        scores_max = np.max(scores, axis=1, keepdims=True)
        scores_shifted = scores - scores_max  # (1, seq_len)

        # Softmax to get attention weights
        exp_scores = np.exp(np.clip(scores_shifted, -500, 500))  # (1, seq_len)
        attn_weights = exp_scores / (np.sum(exp_scores, axis=1, keepdims=True) + 1e-10)  # (1, seq_len)

        # Compute context: context = V @ attn_weights^T -> (hidden_size, 1)
        context = np.dot(V, attn_weights.T)  # (hidden_size, 1)

        # Combine context with query (last hidden state) and project
        # concat: [query; context] -> (2*hidden_size, 1)
        concat_hc = np.vstack([query, context])  # (2*hidden_size, 1)

        # Project back to hidden_size: h_attn = W_attn_out @ concat + b_attn_out
        h_attn = np.dot(self.W_attn_out, concat_hc) + self.b_attn_out  # (hidden_size, 1)

        # Cache for backward pass
        cache = {
            'query': query,
            'H': H,
            'Q': Q,
            'K': K,
            'V': V,
            'scores': scores,
            'scores_shifted': scores_shifted,
            'attn_weights': attn_weights,
            'context': context,
            'concat_hc': concat_hc,
            'h_attn': h_attn,
            'seq_len': seq_len
        }

        # Return attention weights as 1D for interpretability
        return h_attn, attn_weights.flatten(), cache

    def _lstm_cell_forward(
        self,
        xt: Any,
        h_prev: Any,
        c_prev: Any
    ) -> Tuple[Any, Any, Dict[str, Any]]:
        """Single LSTM cell forward pass @zara"""
        np = _ensure_numpy()

        concat = np.vstack([h_prev, xt])

        ft = self._sigmoid(np.dot(self.Wf, concat) + self.bf)
        it = self._sigmoid(np.dot(self.Wi, concat) + self.bi)
        ct_hat = self._tanh(np.dot(self.Wc, concat) + self.bc)
        ot = self._sigmoid(np.dot(self.Wo, concat) + self.bo)

        ct = ft * c_prev + it * ct_hat
        ht = ot * self._tanh(ct)

        cache = {
            'concat': concat,
            'ft': ft, 'it': it, 'ct_hat': ct_hat, 'ot': ot,
            'c_prev': c_prev, 'ct': ct, 'ht': ht,
            'h_prev': h_prev, 'xt': xt
        }

        return ht, ct, cache

    def forward(
        self,
        X: Any,
        training: bool = False
    ) -> Tuple[Any, Optional[Dict[str, Any]]]:
        """Forward pass through sequence @zara"""
        np = _ensure_numpy()

        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))
        caches = []
        hidden_states = []  # Store all hidden states for attention

        for t in range(X.shape[0]):
            xt = X[t:t+1].T
            h, c, cache = self._lstm_cell_forward(xt, h, c)

            if training and self.dropout > 0:
                dropout_mask = (np.random.rand(*h.shape) > self.dropout).astype(float)
                dropout_mask /= (1 - self.dropout)
                h = h * dropout_mask
                cache['dropout_mask'] = dropout_mask

            if training:
                caches.append(cache)

            # Store hidden state (copy to avoid reference issues)
            hidden_states.append(h.copy())

        # Apply attention if enabled
        attn_cache = None
        attn_weights = None
        if self.use_attention and len(hidden_states) > 0:
            # Use final hidden state as query, all states as keys/values
            h_attn, attn_weights, attn_cache = self._compute_attention(h, hidden_states)
            # Use attention-enhanced hidden state for output
            h_for_output = h_attn
        else:
            h_for_output = h

        y_pred = np.dot(self.Wy, h_for_output) + self.by

        if training:
            return y_pred, {
                'caches': caches,
                'final_h': h,
                'hidden_states': hidden_states,
                'attn_cache': attn_cache,
                'attn_weights': attn_weights,
                'h_for_output': h_for_output
            }
        return y_pred, None

    def _lstm_cell_backward(
        self,
        dh_next: Any,
        dc_next: Any,
        cache: Dict[str, Any]
    ) -> Tuple[Any, Any, Dict[str, Any]]:
        """Single LSTM cell backward pass @zara"""
        np = _ensure_numpy()

        concat = cache['concat']
        ft, it, ct_hat, ot = cache['ft'], cache['it'], cache['ct_hat'], cache['ot']
        c_prev, ct = cache['c_prev'], cache['ct']

        dot = dh_next * self._tanh(ct) * ot * (1 - ot)
        dc = dh_next * ot * (1 - self._tanh(ct)**2) + dc_next

        dft = dc * c_prev * ft * (1 - ft)
        dit = dc * ct_hat * it * (1 - it)
        dct_hat = dc * it * (1 - ct_hat**2)

        grads = {}
        grads['Wf'] = np.dot(dft, concat.T)
        grads['Wi'] = np.dot(dit, concat.T)
        grads['Wc'] = np.dot(dct_hat, concat.T)
        grads['Wo'] = np.dot(dot, concat.T)

        grads['bf'] = dft
        grads['bi'] = dit
        grads['bc'] = dct_hat
        grads['bo'] = dot

        dconcat = (
            np.dot(self.Wf.T, dft) +
            np.dot(self.Wi.T, dit) +
            np.dot(self.Wc.T, dct_hat) +
            np.dot(self.Wo.T, dot)
        )

        dh_prev = dconcat[:self.hidden_size, :]
        dc_prev = dc * ft

        return dh_prev, dc_prev, grads

    def _attention_backward(
        self,
        dh_attn: Any,
        attn_cache: Dict[str, Any]
    ) -> Tuple[Any, Dict[str, Any]]:
        """Backward pass through attention mechanism @zara

        Args:
            dh_attn: Gradient w.r.t. attention output (hidden_size, 1)
            attn_cache: Cache from forward attention pass

        Returns:
            dh_final: Gradient w.r.t. final hidden state (for LSTM backprop)
            grads: Dictionary of attention weight gradients
        """
        np = _ensure_numpy()

        # Retrieve cached values
        query = attn_cache['query']  # (hidden_size, 1)
        H = attn_cache['H']  # (hidden_size, seq_len)
        Q = attn_cache['Q']  # (hidden_size, 1)
        K = attn_cache['K']  # (hidden_size, seq_len)
        V = attn_cache['V']  # (hidden_size, seq_len)
        attn_weights = attn_cache['attn_weights']  # (1, seq_len)
        context = attn_cache['context']  # (hidden_size, 1)
        concat_hc = attn_cache['concat_hc']  # (2*hidden_size, 1)
        seq_len = attn_cache['seq_len']

        grads = {}

        # 1. Backprop through output projection: h_attn = W_attn_out @ concat_hc + b_attn_out
        # dh_attn: (hidden_size, 1)
        grads['W_attn_out'] = np.dot(dh_attn, concat_hc.T)  # (hidden_size, 2*hidden_size)
        grads['b_attn_out'] = dh_attn.copy()  # (hidden_size, 1)

        d_concat_hc = np.dot(self.W_attn_out.T, dh_attn)  # (2*hidden_size, 1)

        # Split d_concat_hc into d_query and d_context
        d_query_from_concat = d_concat_hc[:self.hidden_size, :]  # (hidden_size, 1)
        d_context = d_concat_hc[self.hidden_size:, :]  # (hidden_size, 1)

        # 2. Backprop through context = V @ attn_weights^T
        # d_context: (hidden_size, 1), attn_weights: (1, seq_len), V: (hidden_size, seq_len)
        d_V = np.dot(d_context, attn_weights)  # (hidden_size, seq_len)
        d_attn_weights = np.dot(V.T, d_context).T  # (1, seq_len)

        # 3. Backprop through softmax
        # attn_weights: (1, seq_len), d_attn_weights: (1, seq_len)
        # Softmax backward: d_scores = attn_weights * (d_attn_weights - sum(attn_weights * d_attn_weights))
        sum_term = np.sum(attn_weights * d_attn_weights, axis=1, keepdims=True)
        d_scores = attn_weights * (d_attn_weights - sum_term)  # (1, seq_len)

        # 4. Backprop through scaling: scores = Q^T @ K / sqrt(d_k)
        d_k = self.hidden_size
        d_scores_unscaled = d_scores / np.sqrt(d_k)  # (1, seq_len)

        # scores = Q^T @ K -> d_Q = K @ d_scores^T, d_K = Q @ d_scores
        d_Q = np.dot(K, d_scores_unscaled.T)  # (hidden_size, 1)
        d_K = np.dot(Q, d_scores_unscaled)  # (hidden_size, seq_len)

        # 5. Backprop through Q = W_query @ query
        grads['W_query'] = np.dot(d_Q, query.T)  # (hidden_size, hidden_size)
        d_query_from_Q = np.dot(self.W_query.T, d_Q)  # (hidden_size, 1)

        # 6. Backprop through K = W_key @ H
        grads['W_key'] = np.dot(d_K, H.T)  # (hidden_size, hidden_size)
        d_H_from_K = np.dot(self.W_key.T, d_K)  # (hidden_size, seq_len)

        # 7. Backprop through V = W_value @ H
        grads['W_value'] = np.dot(d_V, H.T)  # (hidden_size, hidden_size)
        d_H_from_V = np.dot(self.W_value.T, d_V)  # (hidden_size, seq_len)

        # 8. Total gradient on H (all hidden states)
        d_H = d_H_from_K + d_H_from_V  # (hidden_size, seq_len)

        # 9. Total gradient on query (final hidden state)
        # query receives gradients from: concat (d_query_from_concat), Q projection (d_query_from_Q)
        # and as the last column of H (d_H[:, -1])
        d_query_total = d_query_from_concat + d_query_from_Q + d_H[:, -1:].reshape(-1, 1)

        return d_query_total, grads

    def backward(self, dy: Any, cache: Dict[str, Any]) -> Dict[str, Any]:
        """Backward pass through sequence (BPTT) @zara"""
        np = _ensure_numpy()

        caches = cache['caches']
        final_h = cache['final_h']
        h_for_output = cache.get('h_for_output', final_h)
        attn_cache = cache.get('attn_cache', None)

        # Gradient from output layer
        dh_out = np.dot(self.Wy.T, dy)
        grads_Wy = np.dot(dy, h_for_output.T)
        grads_by = dy

        grads_accum = {
            'Wf': np.zeros_like(self.Wf), 'Wi': np.zeros_like(self.Wi),
            'Wc': np.zeros_like(self.Wc), 'Wo': np.zeros_like(self.Wo),
            'bf': np.zeros_like(self.bf), 'bi': np.zeros_like(self.bi),
            'bc': np.zeros_like(self.bc), 'bo': np.zeros_like(self.bo)
        }

        # If attention was used, backprop through attention first
        if self.use_attention and attn_cache is not None:
            # Backprop through attention mechanism
            dh_final, attn_grads = self._attention_backward(dh_out, attn_cache)
            dh = dh_final

            # Add attention gradients to accumulator
            for key in attn_grads:
                grads_accum[key] = attn_grads[key]
        else:
            dh = dh_out

        dc = np.zeros((self.hidden_size, 1))

        for t in reversed(range(len(caches))):
            cache_t = caches[t]

            if 'dropout_mask' in cache_t:
                dh = dh * cache_t['dropout_mask']

            dh_prev, dc_prev, grads = self._lstm_cell_backward(dh, dc, cache_t)

            for key in ['Wf', 'Wi', 'Wc', 'Wo', 'bf', 'bi', 'bc', 'bo']:
                grads_accum[key] += grads[key]

            dh = dh_prev
            dc = dc_prev

        grads_accum['Wy'] = grads_Wy
        grads_accum['by'] = grads_by

        max_norm = 5.0
        total_norm = np.sqrt(sum(np.sum(g**2) for g in grads_accum.values()))
        if total_norm > max_norm:
            for key in grads_accum:
                grads_accum[key] *= max_norm / total_norm

        return grads_accum

    def _update_weights_adam(self, grads: Dict[str, Any], t: int):
        """Update weights using Adam optimizer @zara"""
        np = _ensure_numpy()

        for key in grads:
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)

            m_hat = self.m[key] / (1 - self.beta1 ** t)
            v_hat = self.v[key] / (1 - self.beta2 ** t)

            param = getattr(self, key)
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            setattr(self, key, param)

    async def train(
        self,
        X_sequences: List[Any],
        y_targets: List[Any],
        epochs: int = 100,
        batch_size: int = 16,
        validation_split: float = 0.2,
        early_stopping_patience: int = 10,
        checkpoint_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Train Multi-Output LSTM with backpropagation @zara

        Args:
            X_sequences: List of input sequences, each shape (seq_len, input_size)
            y_targets: List of targets - List[List[float]] for multi-output
            epochs: Maximum training epochs
            batch_size: Mini-batch size
            validation_split: Fraction of data for validation
            early_stopping_patience: Stop if no improvement for this many epochs
            checkpoint_callback: Optional callback for checkpointing

        Returns:
            Training results dict with accuracy, losses, etc.
        """
        np = _ensure_numpy()

        _LOGGER.info(
            f"Training: {len(X_sequences)} samples, {epochs} epochs, "
            f"{self.num_outputs} outputs"
        )

        def _to_target_array(y):
            """Convert target to (num_outputs, 1) array @zara"""
            if isinstance(y, (int, float)):
                return np.array([[float(y)]] * self.num_outputs)
            elif isinstance(y, (list, tuple)):
                # List of values - one per output
                arr = np.array([[float(v)] for v in y])
                if arr.shape[0] != self.num_outputs:
                    # Pad or truncate
                    if arr.shape[0] < self.num_outputs:
                        arr = np.vstack([arr, np.zeros((self.num_outputs - arr.shape[0], 1))])
                    else:
                        arr = arr[:self.num_outputs]
                return arr
            else:
                return np.array([[0.0]] * self.num_outputs)

        n_samples = len(X_sequences)
        n_val = int(n_samples * validation_split)
        n_train = n_samples - n_val

        # Ensure at least 1 validation sample for meaningful early stopping
        if n_val < 1 and n_samples >= 2:
            n_val = 1
            n_train = n_samples - 1
        elif n_samples < 2:
            # Not enough samples for validation - use all for training
            n_val = 0
            n_train = n_samples

        indices = np.random.permutation(n_samples)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]

        X_train = [X_sequences[i] for i in train_idx]
        y_train = [y_targets[i] for i in train_idx]
        X_val = [X_sequences[i] for i in val_idx]
        y_val = [y_targets[i] for i in val_idx]

        best_val_loss = float('inf')
        patience_counter = 0
        use_early_stopping = len(X_val) > 0  # Only use early stopping if we have validation data
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(1, epochs + 1):
            train_indices = np.random.permutation(n_train)
            epoch_loss = 0.0

            for batch_start in range(0, n_train, batch_size):
                batch_end = min(batch_start + batch_size, n_train)
                batch_indices = train_indices[batch_start:batch_end]

                batch_loss = 0.0
                batch_grads = None

                for idx in batch_indices:
                    X = np.array(X_train[idx])
                    y_true = _to_target_array(y_train[idx])

                    y_pred, cache = self.forward(X, training=True)
                    loss = ((y_pred - y_true) ** 2).mean()
                    batch_loss += loss

                    dy = 2 * (y_pred - y_true) / y_true.size
                    grads = self.backward(dy, cache)

                    if batch_grads is None:
                        batch_grads = grads
                    else:
                        for key in batch_grads:
                            batch_grads[key] += grads[key]

                if batch_grads is None:
                    # All samples in batch failed - skip weight update
                    continue

                for key in batch_grads:
                    batch_grads[key] /= len(batch_indices)

                self._update_weights_adam(batch_grads, epoch)
                epoch_loss += batch_loss

            train_loss = epoch_loss / n_train if n_train > 0 else 0.0

            val_loss = 0.0
            if use_early_stopping:
                for i in range(len(X_val)):
                    X = np.array(X_val[i])
                    y_true = _to_target_array(y_val[i])
                    y_pred, _ = self.forward(X, training=False)
                    val_loss += ((y_pred - y_true) ** 2).mean()
                val_loss /= len(X_val)

            history['train_loss'].append(float(train_loss))
            history['val_loss'].append(float(val_loss))

            if epoch % 10 == 0:
                _LOGGER.info(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")

            if use_early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    _LOGGER.info(f"Early stopping at epoch {epoch}")
                    break

            if checkpoint_callback and epoch % 10 == 0:
                await checkpoint_callback(epoch, self.get_weights())

            if epoch % 5 == 0:
                await asyncio.sleep(0)

        # Calculate R2 score for total prediction (sum of outputs)
        y_pred_totals = []
        y_true_totals = []
        for i, X in enumerate(X_val):
            y_pred, _ = self.forward(np.array(X), training=False)
            y_pred_totals.append(float(np.sum(y_pred)))

            y_true_arr = _to_target_array(y_val[i])
            y_true_totals.append(float(np.sum(y_true_arr)))

        r2_score = 0.0
        rmse = 0.0
        if len(y_true_totals) > 0:
            y_true_arr = np.array(y_true_totals)
            y_pred_arr = np.array(y_pred_totals)
            ss_res = np.sum((y_true_arr - y_pred_arr) ** 2)
            ss_tot = np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)
            r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            # RMSE: Root Mean Squared Error in kWh
            rmse = float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))

        _LOGGER.info(f"Training complete: R2={r2_score:.3f}, RMSE={rmse:.3f} kWh, outputs={self.num_outputs}")

        return {
            'success': True,
            'accuracy': float(r2_score),
            'rmse': rmse,
            'final_train_loss': float(history['train_loss'][-1]),
            'final_val_loss': float(history['val_loss'][-1]),
            'best_val_loss': float(best_val_loss),
            'epochs_trained': epoch,
            'num_outputs': self.num_outputs,
            'history': history
        }

    def predict(self, X_sequence: Any) -> List[float]:
        """Predict production for single sequence - returns all outputs @zara

        Args:
            X_sequence: Input sequence of shape (seq_len, input_size)

        Returns:
            List of predictions, one per output neuron (one per panel group)
        """
        np = _ensure_numpy()
        X_arr = np.array(X_sequence)

        if X_arr.ndim == 3:
            if X_arr.shape[0] != 1:
                raise ValueError(f"Batch not supported, got shape {X_arr.shape}")
            X_arr = X_arr.squeeze(0)

        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D, got {X_arr.ndim}D shape {X_arr.shape}")

        if X_arr.shape[0] != self.sequence_length:
            raise ValueError(
                f"Sequence length: expected {self.sequence_length}, got {X_arr.shape[0]}"
            )

        if X_arr.shape[1] != self.input_size:
            raise ValueError(
                f"Features: expected {self.input_size}, got {X_arr.shape[1]}"
            )

        y_pred, _ = self.forward(X_arr, training=False)
        # y_pred has shape (num_outputs, 1) - flatten to list
        return [float(y_pred[i, 0]) for i in range(self.num_outputs)]

    def predict_total(self, X_sequence: Any) -> float:
        """Predict total production (sum of all group outputs) @zara

        Args:
            X_sequence: Input sequence of shape (seq_len, input_size)

        Returns:
            Sum of all output predictions
        """
        predictions = self.predict(X_sequence)
        return sum(predictions)

    def get_weights(self) -> Dict[str, Any]:
        """Export weights for persistence @zara"""
        weights = {
            'Wf': self.Wf.tolist(), 'Wi': self.Wi.tolist(),
            'Wc': self.Wc.tolist(), 'Wo': self.Wo.tolist(),
            'bf': self.bf.tolist(), 'bi': self.bi.tolist(),
            'bc': self.bc.tolist(), 'bo': self.bo.tolist(),
            'Wy': self.Wy.tolist(), 'by': self.by.tolist(),
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'sequence_length': self.sequence_length,
            'num_outputs': self.num_outputs,
            'has_attention': self.use_attention,
        }

        # Add attention weights if attention is enabled
        if self.use_attention:
            weights['W_query'] = self.W_query.tolist()
            weights['W_key'] = self.W_key.tolist()
            weights['W_value'] = self.W_value.tolist()
            weights['W_attn_out'] = self.W_attn_out.tolist()
            weights['b_attn_out'] = self.b_attn_out.tolist()

        return weights

    def set_weights(self, weights: Dict[str, Any]):
        """Load weights from persistence @zara"""
        np = _ensure_numpy()

        self.Wf = np.array(weights['Wf'])
        self.Wi = np.array(weights['Wi'])
        self.Wc = np.array(weights['Wc'])
        self.Wo = np.array(weights['Wo'])
        self.bf = np.array(weights['bf'])
        self.bi = np.array(weights['bi'])
        self.bc = np.array(weights['bc'])
        self.bo = np.array(weights['bo'])
        self.Wy = np.array(weights['Wy'])
        self.by = np.array(weights['by'])

        self.input_size = weights.get('input_size', self.input_size)
        self.hidden_size = weights.get('hidden_size', self.hidden_size)
        self.sequence_length = weights.get('sequence_length', self.sequence_length)
        self.num_outputs = weights.get('num_outputs', self.Wy.shape[0])

        # Handle attention weights - check if saved weights have attention
        has_saved_attention = weights.get('has_attention', False)

        if has_saved_attention and self.use_attention:
            # Load attention weights from saved model
            self.W_query = np.array(weights['W_query'])
            self.W_key = np.array(weights['W_key'])
            self.W_value = np.array(weights['W_value'])
            self.W_attn_out = np.array(weights['W_attn_out'])
            self.b_attn_out = np.array(weights['b_attn_out'])

            # Reinitialize Adam states for attention weights
            self.m['W_query'] = np.zeros_like(self.W_query)
            self.m['W_key'] = np.zeros_like(self.W_key)
            self.m['W_value'] = np.zeros_like(self.W_value)
            self.m['W_attn_out'] = np.zeros_like(self.W_attn_out)
            self.m['b_attn_out'] = np.zeros_like(self.b_attn_out)

            self.v['W_query'] = np.zeros_like(self.W_query)
            self.v['W_key'] = np.zeros_like(self.W_key)
            self.v['W_value'] = np.zeros_like(self.W_value)
            self.v['W_attn_out'] = np.zeros_like(self.W_attn_out)
            self.v['b_attn_out'] = np.zeros_like(self.b_attn_out)

            _LOGGER.info(f"Weights loaded: {self.num_outputs} outputs, attention=True")
        elif self.use_attention and not has_saved_attention:
            # Current model uses attention but loaded weights don't have it
            # Keep the randomly initialized attention weights
            _LOGGER.warning(
                f"Loaded legacy weights without attention. "
                f"Attention weights remain randomly initialized."
            )
            _LOGGER.info(f"Weights loaded: {self.num_outputs} outputs, attention=True (new)")
        else:
            _LOGGER.info(f"Weights loaded: {self.num_outputs} outputs")

    def get_model_size_kb(self) -> float:
        """Calculate model size in KB @zara"""
        total_params = (
            self.Wf.size + self.Wi.size + self.Wc.size + self.Wo.size +
            self.bf.size + self.bi.size + self.bc.size + self.bo.size +
            self.Wy.size + self.by.size
        )

        # Add attention parameters if attention is enabled
        if self.use_attention:
            total_params += (
                self.W_query.size + self.W_key.size + self.W_value.size +
                self.W_attn_out.size + self.b_attn_out.size
            )

        return (total_params * 8) / 1024


# Test code for attention mechanism
if __name__ == "__main__":
    import sys

    np = _ensure_numpy()
    np.random.seed(42)

    print("=" * 60)
    print("TinyLSTM Attention Mechanism Tests")
    print("=" * 60)

    all_passed = True

    # Test 1: LSTM without attention (backward compatibility)
    print("\n[Test 1] LSTM without attention (backward compatibility)")
    try:
        lstm_no_attn = TinyLSTM(
            input_size=20,
            hidden_size=32,
            sequence_length=24,
            num_outputs=1,
            use_attention=False
        )
        X_test = np.random.randn(24, 20)
        y_pred, _ = lstm_no_attn.forward(X_test, training=False)
        assert y_pred.shape == (1, 1), f"Expected (1, 1), got {y_pred.shape}"
        assert not np.isnan(y_pred).any(), "Output contains NaN"
        print(f"  PASSED - Output shape: {y_pred.shape}, value: {y_pred[0, 0]:.4f}")
    except Exception as e:
        print(f"  FAILED - {e}")
        all_passed = False

    # Test 2: LSTM with attention (forward pass)
    print("\n[Test 2] LSTM with attention (forward pass)")
    try:
        lstm_attn = TinyLSTM(
            input_size=20,
            hidden_size=32,
            sequence_length=24,
            num_outputs=1,
            use_attention=True
        )
        X_test = np.random.randn(24, 20)
        y_pred, cache = lstm_attn.forward(X_test, training=True)
        assert y_pred.shape == (1, 1), f"Expected (1, 1), got {y_pred.shape}"
        assert not np.isnan(y_pred).any(), "Output contains NaN"
        assert cache['attn_weights'] is not None, "Attention weights not in cache"
        assert cache['attn_cache'] is not None, "Attention cache is None"
        print(f"  PASSED - Output shape: {y_pred.shape}, value: {y_pred[0, 0]:.4f}")
    except Exception as e:
        print(f"  FAILED - {e}")
        all_passed = False

    # Test 3: Attention weights sum to 1
    print("\n[Test 3] Attention weights sum to 1")
    try:
        lstm_attn = TinyLSTM(
            input_size=20,
            hidden_size=32,
            sequence_length=24,
            num_outputs=1,
            use_attention=True
        )
        X_test = np.random.randn(24, 20)
        y_pred, cache = lstm_attn.forward(X_test, training=True)
        attn_weights = cache['attn_weights']
        attn_sum = np.sum(attn_weights)
        assert attn_weights.shape == (24,), f"Expected (24,), got {attn_weights.shape}"
        assert np.abs(attn_sum - 1.0) < 1e-6, f"Attention sum is {attn_sum}, expected 1.0"
        assert np.all(attn_weights >= 0), "Attention weights contain negative values"
        print(f"  PASSED - Shape: {attn_weights.shape}, Sum: {attn_sum:.6f}")
        print(f"  Attention distribution (first 5): {attn_weights[:5]}")
    except Exception as e:
        print(f"  FAILED - {e}")
        all_passed = False

    # Test 4: Backward pass produces no NaN values
    print("\n[Test 4] Backward pass produces no NaN values")
    try:
        lstm_attn = TinyLSTM(
            input_size=20,
            hidden_size=32,
            sequence_length=24,
            num_outputs=1,
            use_attention=True
        )
        X_test = np.random.randn(24, 20)
        y_target = np.array([[0.5]])

        y_pred, cache = lstm_attn.forward(X_test, training=True)
        dy = 2 * (y_pred - y_target)
        grads = lstm_attn.backward(dy, cache)

        has_nan = False
        for key, grad in grads.items():
            if np.isnan(grad).any():
                print(f"  NaN found in gradient: {key}")
                has_nan = True

        assert not has_nan, "Gradients contain NaN values"

        # Check attention gradients are present
        assert 'W_query' in grads, "W_query gradient missing"
        assert 'W_key' in grads, "W_key gradient missing"
        assert 'W_value' in grads, "W_value gradient missing"
        assert 'W_attn_out' in grads, "W_attn_out gradient missing"
        assert 'b_attn_out' in grads, "b_attn_out gradient missing"

        print(f"  PASSED - All {len(grads)} gradients computed without NaN")
        print(f"  Gradient keys: {list(grads.keys())}")
    except Exception as e:
        print(f"  FAILED - {e}")
        all_passed = False

    # Test 5: Weights can be saved and loaded
    print("\n[Test 5] Weights can be saved and loaded")
    try:
        # Create model with attention
        lstm_orig = TinyLSTM(
            input_size=20,
            hidden_size=32,
            sequence_length=24,
            num_outputs=1,
            use_attention=True
        )

        # Get original prediction
        X_test = np.random.randn(24, 20)
        y_orig, _ = lstm_orig.forward(X_test, training=False)

        # Save weights
        weights = lstm_orig.get_weights()
        assert weights['has_attention'] is True, "has_attention flag not set"
        assert 'W_query' in weights, "W_query not in saved weights"

        # Create new model and load weights
        lstm_loaded = TinyLSTM(
            input_size=20,
            hidden_size=32,
            sequence_length=24,
            num_outputs=1,
            use_attention=True
        )
        lstm_loaded.set_weights(weights)

        # Compare predictions
        y_loaded, _ = lstm_loaded.forward(X_test, training=False)
        diff = np.abs(y_orig - y_loaded).max()
        assert diff < 1e-10, f"Predictions differ by {diff}"

        print(f"  PASSED - Weights saved/loaded, prediction diff: {diff:.2e}")
    except Exception as e:
        print(f"  FAILED - {e}")
        all_passed = False

    # Test 6: Legacy weights (without attention) load gracefully
    print("\n[Test 6] Legacy weights (without attention) load gracefully")
    try:
        # Create model without attention
        lstm_legacy = TinyLSTM(
            input_size=20,
            hidden_size=32,
            sequence_length=24,
            num_outputs=1,
            use_attention=False
        )
        legacy_weights = lstm_legacy.get_weights()
        assert legacy_weights.get('has_attention') is False, "has_attention should be False"

        # Create model WITH attention and load legacy weights
        lstm_new = TinyLSTM(
            input_size=20,
            hidden_size=32,
            sequence_length=24,
            num_outputs=1,
            use_attention=True
        )
        lstm_new.set_weights(legacy_weights)

        # Should still work (attention weights randomly initialized)
        X_test = np.random.randn(24, 20)
        y_pred, _ = lstm_new.forward(X_test, training=False)
        assert not np.isnan(y_pred).any(), "Output contains NaN"

        print(f"  PASSED - Legacy weights loaded, model still works")
    except Exception as e:
        print(f"  FAILED - {e}")
        all_passed = False

    # Test 7: Model size calculation includes attention
    print("\n[Test 7] Model size calculation includes attention")
    try:
        lstm_no_attn = TinyLSTM(
            input_size=20,
            hidden_size=32,
            sequence_length=24,
            num_outputs=1,
            use_attention=False
        )
        lstm_with_attn = TinyLSTM(
            input_size=20,
            hidden_size=32,
            sequence_length=24,
            num_outputs=1,
            use_attention=True
        )

        size_no_attn = lstm_no_attn.get_model_size_kb()
        size_with_attn = lstm_with_attn.get_model_size_kb()

        assert size_with_attn > size_no_attn, "Model with attention should be larger"

        # Calculate expected additional params:
        # W_query: 32x32 = 1024
        # W_key: 32x32 = 1024
        # W_value: 32x32 = 1024
        # W_attn_out: 32x64 = 2048
        # b_attn_out: 32 = 32
        # Total: 5152 params * 8 bytes / 1024 = ~40.25 KB additional
        expected_diff = (5152 * 8) / 1024
        actual_diff = size_with_attn - size_no_attn
        assert np.abs(actual_diff - expected_diff) < 0.1, f"Size diff mismatch"

        print(f"  PASSED - No attention: {size_no_attn:.2f} KB, With attention: {size_with_attn:.2f} KB")
        print(f"  Attention adds {actual_diff:.2f} KB (expected ~{expected_diff:.2f} KB)")
    except Exception as e:
        print(f"  FAILED - {e}")
        all_passed = False

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED!")
        sys.exit(1)
