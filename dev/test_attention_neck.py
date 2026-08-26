"""Probas do neck de atención local engadido para a comparación arquitectónica do artigo.

O obxectivo destas probas non é validar a calidade do modelo: é garantir que a variante
`neck_type="attention"` non rompe o comportamento por defecto e que non produce NaN, que
é o fallo que apareceu na primeira implementación (o forward saía limpo pero o gradiente
non, porque unha fila de atención totalmente enmascarada devolve NaN no softmax).

    python -m unittest dev.test_attention_neck
"""

import unittest

import torch

from src.temporalmaxer_continuous import LocalSelfAttentionBlock, TemporalMaxerContinuous

BATCH, LENGTH, DIM = 2, 240, 512


def build(neck: str, seed: int = 7, **kwargs) -> TemporalMaxerContinuous:
    torch.manual_seed(seed)
    return TemporalMaxerContinuous(
        input_dim=DIM, hidden_dim=128, pyramid_levels=6, neck_type=neck, **kwargs
    )


def collect(output, accumulator=None):
    accumulator = [] if accumulator is None else accumulator
    if torch.is_tensor(output):
        accumulator.append(output.flatten())
    elif isinstance(output, dict):
        for value in output.values():
            collect(value, accumulator)
    elif isinstance(output, (list, tuple)):
        for value in output:
            collect(value, accumulator)
    return accumulator


def flatten(output) -> torch.Tensor:
    return torch.cat(collect(output))


class AttentionNeckTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.features = torch.randn(BATCH, LENGTH, DIM)
        self.mask = torch.ones(BATCH, LENGTH, dtype=torch.bool)
        self.mask[1, 180:] = False

    def test_default_is_unchanged(self):
        """O modelo por defecto segue sendo TemporalMaxer puro."""
        model = build("maxpool")
        self.assertEqual(model.neck_type, "maxpool")
        self.assertIsNone(model.classification_neck)
        self.assertIsNone(model.regression_neck)

    def test_default_is_reproducible(self):
        first, second = build("maxpool").eval(), build("maxpool").eval()
        with torch.no_grad():
            left = flatten(first(self.features, self.mask))
            right = flatten(second(self.features, self.mask))
        self.assertTrue(torch.allclose(left, right))

    def test_attention_output_is_finite(self):
        model = build("attention").eval()
        with torch.no_grad():
            values = flatten(model(self.features, self.mask))
        self.assertTrue(torch.isfinite(values).all())

    def test_attention_keeps_the_output_contract(self):
        """Mesmas claves e mesmas formas cá variante por defecto."""
        maxpool, attention = build("maxpool").eval(), build("attention").eval()
        with torch.no_grad():
            expected = maxpool(self.features, self.mask)
            actual = attention(self.features, self.mask)
        self.assertEqual(sorted(expected), sorted(actual))
        for key in expected:
            for left, right in zip(collect(expected[key]), collect(actual[key])):
                self.assertEqual(left.shape, right.shape)

    def test_gradients_are_finite(self):
        """O fallo orixinal: forward limpo pero gradiente con NaN."""
        model = build("attention")
        model.train()
        flatten(model(self.features, self.mask)).square().mean().backward()
        neck_total = 0.0
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            self.assertTrue(torch.isfinite(parameter.grad).all(), f"grad non finito en {name}")
            if "neck" in name:
                neck_total += parameter.grad.abs().sum().item()
        self.assertGreater(neck_total, 0.0, "o neck de atención non recibe gradiente")

    def test_degenerate_mask(self):
        """Unha secuencia cun só bin válido non pode xerar NaN."""
        model = build("attention")
        model.train()
        mask = torch.zeros(1, LENGTH, dtype=torch.bool)
        mask[0, 0] = True
        flatten(model(self.features[:1], mask)).square().mean().backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all(), f"NaN en {name}")

    def test_window_is_respected(self):
        """Fóra da ventá non pode haber influencia."""
        block = LocalSelfAttentionBlock(channels=16, heads=2, window=5, dropout=0.0).eval()
        mask = torch.ones(1, 1, 40, dtype=torch.bool)
        blocked = block._attention_mask(mask)[0]
        self.assertFalse(blocked[10, 10].item())          # a diagonal sempre aberta
        self.assertFalse(blocked[10, 12].item())          # dentro da ventá (|d|=2 <= 2)
        self.assertTrue(blocked[10, 13].item())           # fóra da ventá (|d|=3 > 2)

    def test_rejects_bad_configuration(self):
        for window in (0, 20, -3):
            with self.assertRaises(ValueError):
                build("attention", attention_window=window)
        with self.assertRaises(ValueError):
            build("not-a-neck")


if __name__ == "__main__":
    unittest.main()
