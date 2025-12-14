"""TinyML LSTM - NumPy-only implementation for Home Assistant V12.0.0 @zara

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Copyright (C) 2025 Zara-Toorox
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

_LOGGER = logging.getLogger(__name__)

_np: Optional[Any] = None

def _ensure_numpy() -> Any:
    """Lazily imports and returns the NumPy module @zara"""
    global _np
    if _np is None:
        try:
            import numpy as np
            _np = np
        except ImportError as e:
            _LOGGER.error(f"NumPy is required for TinyLSTM: {e}")
            raise ImportError(f"NumPy library is required for TinyLSTM: {e}") from e
    return _np

class TinyLSTM:
    """
    Lightweight LSTM implementation using only NumPy.

    Features:
    - Single LSTM layer (no stacking for simplicity)
    - Adam optimizer (momentum + RMSprop)
    - Gradient clipping (prevent exploding gradients)
    - Early stopping (prevent overfitting)
    - Checkpoint saving (interrupt-safe)
    """

    def __init__(
        self,
        input_size: int = 14,
        hidden_size: int = 32,
        sequence_length: int = 24,
        learning_rate: float = 0.001,
        dropout: float = 0.2
    ):
        """
        Initialize TinyLSTM.

        Args:
            input_size: Number of input features (14: time + weather + astronomy + lag)
            hidden_size: Number of LSTM neurons (32 = good balance)
            sequence_length: Length of input sequences (24 = 24 hours lookback)
            learning_rate: Learning rate for Adam optimizer
            dropout: Dropout rate for regularization
        """
        np = _ensure_numpy()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.dropout = dropout

        limit = np.sqrt(6 / (input_size + hidden_size))
        concat_size = input_size + hidden_size

        self.Wf = np.random.uniform(-limit, limit, (hidden_size, concat_size))
        self.Wi = np.random.uniform(-limit, limit, (hidden_size, concat_size))
        self.Wc = np.random.uniform(-limit, limit, (hidden_size, concat_size))
        self.Wo = np.random.uniform(-limit, limit, (hidden_size, concat_size))

        self.bf = np.ones((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

        limit_out = np.sqrt(6 / (hidden_size + 1))
        self.Wy = np.random.uniform(-limit_out, limit_out, (1, hidden_size))
        self.by = np.zeros((1, 1))

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
            f"sequence={sequence_length}, lr={learning_rate}"
        )

    def _sigmoid(self, x: Any) -> Any:
        """Sigmoid activation with numerical stability @zara"""
        np = _ensure_numpy()

        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))

    def _tanh(self, x: Any) -> Any:
        """Tanh activation with numerical stability @zara"""
        np = _ensure_numpy()
        x_clipped = np.clip(x, -500, 500)
        return np.tanh(x_clipped)

    def _lstm_cell_forward(
        self,
        xt: Any,
        h_prev: Any,
        c_prev: Any
    ) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Single LSTM cell forward pass.

        Gates:
        - Forget gate: ft = sigmoid(Wf @ [h_prev, xt] + bf)
        - Input gate:  it = sigmoid(Wi @ [h_prev, xt] + bi)
        - Cell gate:   ct_hat = tanh(Wc @ [h_prev, xt] + bc)
        - Output gate: ot = sigmoid(Wo @ [h_prev, xt] + bo)

        Updates:
        - Cell state:   ct = ft * c_prev + it * ct_hat
        - Hidden state: ht = ot * tanh(ct)

        Returns:
            ht: Hidden state
            ct: Cell state
            cache: Intermediate values for backprop
        """
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
        """
        Forward pass through entire sequence.

        Args:
            X: Input sequence (sequence_length, input_size) for single sample
            training: If True, apply dropout and return cache for backprop

        Returns:
            y_pred: Prediction (scalar)
            cache: Cache for backprop (only if training=True)
        """
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
        else:
            return y_pred, None

    def _lstm_cell_backward(
        self,
        dh_next: Any,
        dc_next: Any,
        cache: Dict[str, Any]
    ) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Single LSTM cell backward pass (BPTT).

        Returns:
            dh_prev: Gradient w.r.t. previous hidden state
            dc_prev: Gradient w.r.t. previous cell state
            grads: Weight/bias gradients
        """
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

    def backward(
        self,
        dy: Any,
        cache: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Backward pass through entire sequence (BPTT).

        Args:
            dy: Gradient of loss w.r.t. output
            cache: Cached values from forward pass

        Returns:
            grads: Dictionary of all gradients
        """
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

    def _update_weights_adam(
        self,
        grads: Dict[str, Any],
        t: int
    ):
        """
        Update weights using Adam optimizer.

        Args:
            grads: Gradients from backward pass
            t: Current timestep (for bias correction)
        """
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
        y_targets: List[float],
        epochs: int = 100,
        batch_size: int = 16,
        validation_split: float = 0.2,
        early_stopping_patience: int = 10,
        checkpoint_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Train LSTM using BPTT and Adam optimizer.

        Args:
            X_sequences: List of input sequences (each: [seq_len, features])
            y_targets: List of target values (kWh)
            epochs: Number of training epochs
            batch_size: Mini-batch size
            validation_split: Fraction for validation
            early_stopping_patience: Stop if no improvement for N epochs
            checkpoint_callback: Function to call for checkpoints

        Returns:
            training_result: Dict with accuracy, loss, epochs_trained, etc.
        """
        np = _ensure_numpy()

        _LOGGER.info(f"Starting TinyLSTM training: {len(X_sequences)} samples, {epochs} epochs")

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
        training_history = {'train_loss': [], 'val_loss': []}

        for epoch in range(1, epochs + 1):

            train_indices = np.random.permutation(n_train)
            epoch_loss = 0.0
            n_batches = 0

            for batch_start in range(0, n_train, batch_size):
                batch_end = min(batch_start + batch_size, n_train)
                batch_indices = train_indices[batch_start:batch_end]

                batch_loss = 0.0
                batch_grads = None

                for idx in batch_indices:
                    X = np.array(X_train[idx])
                    y_true = np.array([[y_train[idx]]])

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
                n_batches += 1

            train_loss = epoch_loss / n_train

            val_loss = 0.0
            for i in range(len(X_val)):
                X = np.array(X_val[i])
                y_true = np.array([[y_val[i]]])
                y_pred, _ = self.forward(X, training=False)
                val_loss += ((y_pred - y_true) ** 2).mean()
            val_loss /= len(X_val) if len(X_val) > 0 else 1

            training_history['train_loss'].append(float(train_loss))
            training_history['val_loss'].append(float(val_loss))

            if epoch % 10 == 0 or epoch == 1:
                _LOGGER.info(
                    f"Epoch {epoch}/{epochs}: train_loss={train_loss:.4f}, "
                    f"val_loss={val_loss:.4f}"
                )

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

        y_pred_all = []
        for X in X_val:
            X_arr = np.array(X)
            y_pred, _ = self.forward(X_arr, training=False)
            y_pred_all.append(float(y_pred[0, 0]))

        if len(y_val) > 0:
            y_val_arr = np.array(y_val)
            y_pred_arr = np.array(y_pred_all)
            ss_res = np.sum((y_val_arr - y_pred_arr) ** 2)
            ss_tot = np.sum((y_val_arr - np.mean(y_val_arr)) ** 2)
            r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        else:
            r2_score = 0.0

        _LOGGER.info(f"Training complete: R²={r2_score:.3f}, best_val_loss={best_val_loss:.4f}")

        return {
            'success': True,
            'accuracy': float(r2_score),
            'final_train_loss': float(training_history['train_loss'][-1]),
            'final_val_loss': float(training_history['val_loss'][-1]),
            'best_val_loss': float(best_val_loss),
            'epochs_trained': epoch,
            'training_history': training_history
        }

    def predict(self, X_sequence: Any) -> float:
        """Predict for single sequence (inference mode) @zara

        Args:
            X_sequence: Input sequence. Accepts:
                - 2D array (sequence_length, features) - PREFERRED
                - 3D array (1, sequence_length, features) - will be squeezed

        Returns:
            Predicted value (float)

        Raises:
            ValueError: If input has wrong dimensions or shape
        """
        np = _ensure_numpy()
        X_arr = np.array(X_sequence)

        # Defensive: Handle both 2D and 3D inputs
        if X_arr.ndim == 3:
            if X_arr.shape[0] != 1:
                raise ValueError(
                    f"Batch prediction not supported. Expected batch_size=1, got {X_arr.shape[0]}"
                )
            X_arr = X_arr.squeeze(0)  # (1, 24, 14) -> (24, 14)
            _LOGGER.debug("Squeezed 3D input to 2D for LSTM prediction")

        if X_arr.ndim != 2:
            raise ValueError(
                f"Expected 2D sequence (seq_len, features), got {X_arr.ndim}D with shape {X_arr.shape}"
            )

        if X_arr.shape[0] != self.sequence_length:
            raise ValueError(
                f"Sequence length mismatch: expected {self.sequence_length}, got {X_arr.shape[0]}"
            )

        if X_arr.shape[1] != self.input_size:
            raise ValueError(
                f"Feature count mismatch: expected {self.input_size}, got {X_arr.shape[1]}"
            )

        y_pred, _ = self.forward(X_arr, training=False)
        return float(y_pred[0, 0])

    def get_weights(self) -> Dict[str, Any]:
        """Export weights for saving to learned_weights.json @zara"""
        np = _ensure_numpy()

        return {
            'Wf': self.Wf.tolist(), 'Wi': self.Wi.tolist(),
            'Wc': self.Wc.tolist(), 'Wo': self.Wo.tolist(),
            'bf': self.bf.tolist(), 'bi': self.bi.tolist(),
            'bc': self.bc.tolist(), 'bo': self.bo.tolist(),
            'Wy': self.Wy.tolist(), 'by': self.by.tolist(),
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'sequence_length': self.sequence_length
        }

    def set_weights(self, weights: Dict[str, Any]):
        """Load weights from learned_weights.json @zara"""
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

        _LOGGER.info("TinyLSTM weights loaded successfully")

    def get_model_size_kb(self) -> float:
        """Calculate model size in KB (for monitoring) @zara"""
        np = _ensure_numpy()

        total_params = (
            self.Wf.size + self.Wi.size + self.Wc.size + self.Wo.size +
            self.bf.size + self.bi.size + self.bc.size + self.bo.size +
            self.Wy.size + self.by.size
        )

        size_kb = (total_params * 8) / 1024
        return size_kb
