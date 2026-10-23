from app.services.agiqa3k import infer_generator


def test_generator_inference():
    assert infer_generator("AttnGAN_normal_000.jpg") == "AttnGAN"
    assert infer_generator("midjourney_lowstep_000.jpg") == "midjourney"
