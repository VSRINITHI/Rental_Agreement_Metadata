import config
from src.pipeline import Extractor
from src.prompt import build_messages
from src.retrieve import Example, ExampleRetriever

LABELS = {k: "" for k in config.TARGET_KEYS}


def test_retriever_excludes_same_stem_and_returns_most_similar():
    ex = [Example("a", "rent for flat in hyderabad", LABELS), Example("b", "lease of shop in delhi", LABELS),
          Example("c", "rent for flat in hyderabad again", LABELS)]
    r = ExampleRetriever(ex)
    top = r.top_k("rent for flat in hyderabad", 2, exclude_stem="a")
    assert [e.stem for e in top] == ["c", "b"]


def test_build_messages_few_shot_structure():
    labels = {**LABELS, "Aggrement Value": "12,000", "Party One": "X"}
    msgs = build_messages("target", [("doc", labels)])
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert '"Aggrement Value": 12000' in msgs[2]["content"]
    assert "target" in msgs[3]["content"]


class FakeClient:
    def __init__(self):
        self.messages = None

    def extract(self, messages):
        self.messages = messages
        return {k: None for k in config.TARGET_KEYS}


def test_extractor_never_uses_own_label_as_example(settings, monkeypatch):
    import docx
    train = settings.data_dir / "train"
    train.mkdir(parents=True)
    for stem, body in [("one", "monthly rent flat"), ("two", "monthly rent house")]:
        d = docx.Document()
        d.add_paragraph(body)
        d.save(train / f"{stem}.docx")
    (settings.data_dir / "train.csv").write_text(
        "File Name,Aggrement Value,Aggrement Start Date,Aggrement End Date,Renewal Notice (Days),Party One,Party Two\n"
        "one,1,01.01.2020,31.12.2020,30,A,B\ntwo,2,01.01.2020,31.12.2020,30,C,D\n")
    fake = FakeClient()
    monkeypatch.setenv("FEWSHOT_K", "1")
    ex = Extractor(config.get_settings(), client=fake)
    ex.extract_text("monthly rent flat", exclude_stem="one")
    shot = fake.messages[2]["content"]
    assert '"Party One": "C"' in shot  # example came from 'two', never 'one'
