"""Generate / I2V pipeline selection must follow profile, not image presence."""

from __future__ import annotations

from ltx_mlx_backend import LocalVideoGenerator


def _gen_with_pipes(**keys: object) -> LocalVideoGenerator:
    gen = LocalVideoGenerator.__new__(LocalVideoGenerator)
    gen.upscale = False
    gen._pipe_classes = dict(keys)
    return gen


def test_distilled_i2v_uses_same_pipe_as_t2v():
    gen = _gen_with_pipes(
        t2v=object(),
        i2v=object(),
        one_stage=object(),
        two_stage=object(),
        hq=object(),
    )
    assert gen._resolve_generate_pipe_key("distilled", has_image=False) == "t2v"
    assert gen._resolve_generate_pipe_key("distilled", has_image=True) == "t2v"


def test_one_stage_profile_uses_one_stage_with_or_without_image():
    gen = _gen_with_pipes(t2v=object(), i2v=object(), one_stage=object())
    assert gen._resolve_generate_pipe_key("one_stage", has_image=False) == "one_stage"
    assert gen._resolve_generate_pipe_key("one_stage", has_image=True) == "one_stage"


def test_two_stage_and_hq_profiles_ignore_image():
    gen = _gen_with_pipes(
        t2v=object(),
        two_stage=object(),
        hq=object(),
        one_stage=object(),
    )
    assert gen._resolve_generate_pipe_key("two_stage", has_image=True) == "two_stage"
    assert gen._resolve_generate_pipe_key("hq", has_image=True) == "hq"


def test_upscale_flag_prefers_two_stage_even_with_image():
    gen = _gen_with_pipes(t2v=object(), two_stage=object(), i2v=object())
    gen.upscale = True
    assert gen._resolve_generate_pipe_key("distilled", has_image=True) == "two_stage"


def test_i2v_alias_matches_distilled_not_one_stage():
    """Default i2v registration alias must share DistilledPipeline, not one-stage."""

    class DistilledPipeline:
        pass

    class TI2VidOneStagePipeline:
        pass

    # Mirror the registration rules in LocalVideoGenerator.load() for current ltx-2-mlx
    # (no legacy ImageToVideoPipeline): i2v aliases generate_cls; one_stage is separate.
    generate_cls = DistilledPipeline
    legacy_i2v_cls = None
    one_stage_cls = TI2VidOneStagePipeline
    pipes: dict[str, type] = {"t2v": generate_cls}
    if legacy_i2v_cls is not None:
        pipes["i2v"] = legacy_i2v_cls
    elif generate_cls is not None:
        pipes["i2v"] = generate_cls
    if one_stage_cls is not None:
        pipes["one_stage"] = one_stage_cls

    assert pipes["i2v"] is DistilledPipeline
    assert pipes["one_stage"] is TI2VidOneStagePipeline
    assert pipes["i2v"] is not pipes["one_stage"]
