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
        dropout: float = 0.2
    ):
        """Initialize Multi-Output LSTM with given parameters @zara

        Args:
            input_size: Number of input features (default 20 for base+group)
            hidden_size: LSTM hidden layer size
            sequence_length: Number of timesteps in input sequence
            num_outputs: Number of output predictions (1 per panel group)
            learning_rate: Adam optimizer learning rate
            dropout: Dropout rate for regularization
        """
        np = _ensure_numpy()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.num_outputs = num_outputs
        self.learning_rate = learning_rate
        self.dropout = dropout

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

        _LOGGER.info(
            f"TinyLSTM initialized: input={input_size}, hidden={hidden_size}, "
            f"seq={sequence_length}, outputs={num_outputs}"
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

        y_pred = np.dot(self.Wy, h) + self.by

        if training:
            return y_pred, {'caches': caches, 'final_h': h}
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

    def backward(self, dy: Any, cache: Dict[str, Any]) -> Dict[str, Any]:
        """Backward pass through sequence (BPTT) @zara"""
        np = _ensure_numpy()

        caches = cache['caches']
        final_h = cache['final_h']

        dh = np.dot(self.Wy.T, dy)
        grads_Wy = np.dot(dy, final_h.T)
        grads_by = dy

        grads_accum = {
            'Wf': np.zeros_like(self.Wf), 'Wi': np.zeros_like(self.Wi),
            'Wc': np.zeros_like(self.Wc), 'Wo': np.zeros_like(self.Wo),
            'bf': np.zeros_like(self.bf), 'bi': np.zeros_like(self.bi),
            'bc': np.zeros_like(self.bc), 'bo': np.zeros_like(self.bo)
        }

        dc = np.zeros((self.hidden_size, 1))

        for t in reversed(range(len(caches))):
            cache_t = caches[t]

            if 'dropout_mask' in cache_t:
                dh = dh * cache_t['dropout_mask']

            dh_prev, dc_prev, grads = self._lstm_cell_backward(dh, dc, cache_t)

            for key in grads_accum:
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

        indices = np.random.permutation(n_samples)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]

        X_train = [X_sequences[i] for i in train_idx]
        y_train = [y_targets[i] for i in train_idx]
        X_val = [X_sequences[i] for i in val_idx]
        y_val = [y_targets[i] for i in val_idx]

        best_val_loss = float('inf')
        patience_counter = 0
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

                for key in batch_grads:
                    batch_grads[key] /= len(batch_indices)

                self._update_weights_adam(batch_grads, epoch)
                epoch_loss += batch_loss

            train_loss = epoch_loss / n_train

            val_loss = 0.0
            for i in range(len(X_val)):
                X = np.array(X_val[i])
                y_true = _to_target_array(y_val[i])
                y_pred, _ = self.forward(X, training=False)
                val_loss += ((y_pred - y_true) ** 2).mean()
            val_loss /= len(X_val) if len(X_val) > 0 else 1

            history['train_loss'].append(float(train_loss))
            history['val_loss'].append(float(val_loss))

            if epoch % 10 == 0:
                _LOGGER.info(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")

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
        return {
            'Wf': self.Wf.tolist(), 'Wi': self.Wi.tolist(),
            'Wc': self.Wc.tolist(), 'Wo': self.Wo.tolist(),
            'bf': self.bf.tolist(), 'bi': self.bi.tolist(),
            'bc': self.bc.tolist(), 'bo': self.bo.tolist(),
            'Wy': self.Wy.tolist(), 'by': self.by.tolist(),
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'sequence_length': self.sequence_length,
            'num_outputs': self.num_outputs,
        }

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

        _LOGGER.info(f"Weights loaded: {self.num_outputs} outputs")

    def get_model_size_kb(self) -> float:
        """Calculate model size in KB @zara"""
        total_params = (
            self.Wf.size + self.Wi.size + self.Wc.size + self.Wo.size +
            self.bf.size + self.bi.size + self.bc.size + self.bo.size +
            self.Wy.size + self.by.size
        )
        return (total_params * 8) / 1024
