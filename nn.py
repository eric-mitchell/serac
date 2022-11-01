import torch
import torch.nn as nn

import logging
import time

from utils import factorization

LOG = logging.getLogger(__name__)


class FixableDropout(nn.Module):
    def __init__(self, p: float):
        super().__init__()

        self.p = p
        self.mask_cache = {}
        self.seed = 0

    def resample(self, seed=None):
        if seed is None:
            seed = int(time.time() * 1e6)
        self.mask_cache = {}
        self.seed = seed

    def forward(self, x):
        if self.training:
            if x.shape not in self.mask_cache:
                generator = torch.Generator(x.device).manual_seed(self.seed)
                self.mask_cache[x.shape] = torch.bernoulli(
                    torch.full_like(x, 1 - self.p), generator=generator
                ).bool()
                self.should_resample = False

            x = (self.mask_cache[x.shape] * x) / (1 - self.p)

        return x

    def extra_repr(self) -> str:
        return f"p={self.p}"


class ActMLP(nn.Module):
    def __init__(self, hidden_dim, n_hidden):
        super().__init__()

        self.mlp = MLP(1, 1, hidden_dim, n_hidden, init="id")

    def forward(self, x):
        return self.mlp(x.view(-1, 1)).view(x.shape)


class LightIDMLP(nn.Module):
    def __init__(
        self,
        indim: int,
        outdim: int,
        hidden_dim: int,
        n_hidden: int,
        init: str = None,
        act: str = None,
        rank: int = None,
    ):
        super().__init__()
        LOG.info(f"Building LightIDMLP {[indim] + [rank] + [indim]}")
        self.layer1 = nn.Linear(indim, rank)
        self.layer2 = nn.Linear(rank, indim)
        self.layer2.weight.data[:] = 0
        self.layer2.bias = None

    def forward(self, x):
        h = self.layer1(x).relu()
        return x + self.layer2(h)


class IDMLP(nn.Module):
    def __init__(
        self,
        indim: int,
        outdim: int,
        hidden_dim: int,
        n_hidden: int,
        init: str = None,
        act: str = None,
        rank: int = None,
        n_modes: int = None
    ):
        super().__init__()
        LOG.info(f"Building IDMLP ({init}) {[indim] * (n_hidden + 2)}")
        self.layers = nn.ModuleList(
            [
                LRLinear(indim, indim, rank=rank, relu=idx < n_hidden, init=init, n_modes=n_modes)
                for idx in range(n_hidden + 1)
            ]
        )

    def forward(self, x, mode=None):
        for layer in self.layers:
            x = layer(x, mode=mode)

        return x


class LatentIDMLP(nn.Module):
    def __init__(
        self,
        indim: int,
        outdim: int,
        hidden_dim: int,
        n_hidden: int,
        init: str = None,
        act: str = None,
        rank: int = None,
    ):
        super().__init__()
        LOG.info(f"Building Latent IDMLP ({init}) {[indim] * (n_hidden + 2)}")

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(indim, rank))
        for _ in range(n_hidden - 1):
            self.layers.append(nn.Linear(rank, rank))
        self.layers.append(nn.Linear(rank, outdim))

        for layer in self.layers[:-1]:
            nn.init.xavier_normal_(layer.weight.data)

        if init == "id":
            self.layers[-1].weight.data.zero_()
            self.layers[-1].bias.data.zero_()

        self.init = init

    def forward(self, x):
        out = x
        for layer in self.layers[:-1]:
            out = layer(out).relu()

        out = self.layers[-1](out)
        if self.init == "id":
            return out + x
        else:
            return out


class KLinear(nn.Module):
    def __init__(self, inf, outf, pfrac=0.05, symmetric=True, zero_init: bool = True):
        super().__init__()

        self.inf = inf

        in_fact = factorization(inf)
        out_fact = factorization(outf)

        total_params = 0
        self.a, self.b = nn.ParameterList(), nn.ParameterList()
        for (i1, i2), (o1, o2) in zip(reversed(in_fact), reversed(out_fact)):
            new_params = (o1 * i1 + o2 * i2) * (2 if symmetric else 1)
            if (total_params + new_params) / (inf * outf) > pfrac and len(self.a) > 0:
                break
            total_params += new_params

            self.a.append(nn.Parameter(torch.empty(o1, i1)))
            if symmetric:
                self.a.append(nn.Parameter(torch.empty(o2, i2)))

            self.b.append(nn.Parameter(torch.empty(o2, i2)))
            if symmetric:
                self.b.append(nn.Parameter(torch.empty(o1, i1)))

            assert self.a[-1].kron(self.b[-1]).shape == (outf, inf)

        for factor in self.a:
            nn.init.kaiming_normal_(factor.data)
        for factor in self.b:
            if zero_init:
                factor.data.zero_()
            else:
                nn.init.kaiming_normal_(factor.data)

        print(f"Created ({symmetric}) k-layer using {total_params/(outf*inf):.3f} params, {len(self.a)} comps")
        self.bias = nn.Parameter(torch.zeros(outf))

    def forward(self, x):
        assert x.shape[-1] == self.inf, f"Expected input with {self.inf} dimensions, got {x.shape}"
        w = sum([a.kron(b) for a, b in zip(self.a, self.b)]) / (2 * len(self.a) ** 0.5)
        y = w @ x.T
        if self.bias is not None:
            y = y + self.bias
        return y


class LRLinear(nn.Module):
    def __init__(self, inf, outf, rank: int = None, relu=False, init="id", n_modes=None):
        super().__init__()

        mid_dim = min(rank, inf)
        if init == "id":
            self.u = nn.Parameter(torch.zeros(outf, mid_dim))
            self.v = nn.Parameter(torch.randn(mid_dim, inf))
        elif init == "xavier":
            self.u = nn.Parameter(torch.empty(outf, mid_dim))
            self.v = nn.Parameter(torch.empty(mid_dim, inf))
            nn.init.xavier_uniform_(self.u.data, gain=nn.init.calculate_gain("relu"))
            nn.init.xavier_uniform_(self.v.data, gain=1.0)
        else:
            raise ValueError(f"Unrecognized initialization {init}")

        if n_modes is not None:
            self.mode_shift = nn.Embedding(n_modes, outf)
            self.mode_shift.weight.data.zero_()
            self.mode_scale = nn.Embedding(n_modes, outf)
            self.mode_scale.weight.data.fill_(1)

        self.n_modes = n_modes
        self.bias = nn.Parameter(torch.zeros(outf))
        self.inf = inf
        self.init = init

    def forward(self, x, mode=None):
        if mode is not None:
            assert self.n_modes is not None, "Linear got a mode but wasn't initialized for it"
            assert mode < self.n_modes, f"Input mode {mode} outside of range {self.n_modes}"
        assert x.shape[-1] == self.inf, f"Input wrong dim ({x.shape}, {self.inf})"

        pre_act = (self.u @ (self.v @ x.T)).T
        if self.bias is not None:
            pre_act += self.bias

        if mode is not None:
            if not isinstance(mode, torch.Tensor):
                mode = torch.tensor(mode).to(x.device)
            scale, shift = self.mode_scale(mode), self.mode_shift(mode)
            pre_act = pre_act * scale + shift

        # need clamp instead of relu so gradient at 0 isn't 0
        acts = pre_act.clamp(min=0)
        if self.init == "id":
            return acts + x
        else:
            return acts


class MLP(nn.Module):
    def __init__(
        self,
        indim: int,
        outdim: int,
        hidden_dim: int,
        n_hidden: int,
        init: str = "xavier_uniform",
        act: str = "relu",
        rank: int = None,
    ):
        super().__init__()

        self.init = init

        if act == "relu":
            self.act = nn.ReLU()
        elif act == "learned":
            self.act = ActMLP(10, 1)
        else:
            raise ValueError(f"Unrecognized activation function '{act}'")

        if hidden_dim is None:
            hidden_dim = outdim * 2

        if init.startswith("id") and outdim != indim:
            LOG.info(f"Overwriting outdim ({outdim}) to be indim ({indim})")
            outdim = indim

        if init == "id":
            old_hidden_dim = hidden_dim
            if hidden_dim < indim * 2:
                hidden_dim = indim * 2

            if hidden_dim % indim != 0:
                hidden_dim += hidden_dim % indim

            if old_hidden_dim != hidden_dim:
                LOG.info(
                    f"Overwriting hidden dim ({old_hidden_dim}) to be {hidden_dim}"
                )

        if init == "id_alpha":
            self.alpha = nn.Parameter(torch.zeros(1, outdim))

        dims = [indim] + [hidden_dim] * n_hidden + [outdim]
        LOG.info(f"Building ({init}) MLP: {dims} (rank {rank})")

        layers = []
        for idx, (ind, outd) in enumerate(zip(dims[:-1], dims[1:])):
            if rank is None:
                layers.append(nn.Linear(ind, outd))
            else:
                layers.append(LRLinear(ind, outd, rank=rank))
            if idx < n_hidden:
                layers.append(self.act)

        if rank is None:
            if init == "id":
                if n_hidden > 0:
                    layers[0].weight.data = torch.eye(indim).repeat(
                        hidden_dim // indim, 1
                    )
                    layers[0].weight.data[hidden_dim // 2:] *= -1
                    layers[-1].weight.data = torch.eye(outdim).repeat(
                        1, hidden_dim // outdim
                    )
                    layers[-1].weight.data[:, hidden_dim // 2:] *= -1
                    layers[-1].weight.data /= (hidden_dim // indim) / 2.0

            for layer in layers:
                if isinstance(layer, nn.Linear):
                    if init == "ortho":
                        nn.init.orthogonal_(layer.weight)
                    elif init == "id":
                        if layer.weight.shape[0] == layer.weight.shape[1]:
                            layer.weight.data = torch.eye(hidden_dim)
                    else:
                        gain = 3 ** 0.5 if (layer is layers[-1]) else 1.0
                        nn.init.xavier_uniform_(layer.weight, gain=gain)

                    layer.bias.data[:] = 0

        layers[-1].bias = None
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        if self.init == "id_alpha":
            return x + self.alpha * self.mlp(x)
        else:
            return self.mlp(x)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )
    m0 = MLP(1000, 1000, 1500, 3)
    m1 = MLP(1000, 1000, 1500, 3, init="id")
    m2 = MLP(1000, 1000, 1500, 3, init="id_alpha")
    m3 = MLP(1000, 1000, 1500, 3, init="ortho", act="learned")

    x = 0.01 * torch.randn(999, 1000)

    y0 = m0(x)
    y1 = m1(x)
    y2 = m2(x)
    y3 = m3(x)

    print("y0", (y0 - x).abs().max())
    print("y1", (y1 - x).abs().max())
    print("y2", (y2 - x).abs().max())
    print("y3", (y3 - x).abs().max())

    assert not torch.allclose(y0, x)
    assert torch.allclose(y1, x)
    assert torch.allclose(y2, x)
    assert not torch.allclose(y3, x)
    import pdb; pdb.set_trace()  # fmt: skip
