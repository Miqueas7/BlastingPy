import blastingpy


def test_version_and_core():
    assert blastingpy.__version__ == "0.1.0"
    info = blastingpy.MethodInfo("X", blastingpy.Verification.Secondary, "src")
    assert info.method_id == "X" and info.verification == blastingpy.Verification.Secondary
