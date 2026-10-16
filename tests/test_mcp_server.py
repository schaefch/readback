from readback import toolbox
from readback.mcp_server import mcp


async def test_mcp_server_exposes_review_and_pipeline_tools():
    names = {t.name for t in await mcp.list_tools()}
    assert toolbox.REVIEW_TOOL_NAMES | {"run_qa"} <= names


async def test_compare_texts_tool():
    result = toolbox.compare_texts(
        "Das kostet 2,95 €.", "Das kostet zwei Euro fünfundneunzig.", "de"
    )
    assert result == {"wer": 0.0, "cer": 0.0, "differences": []}
    assert toolbox.check_homophones("tale", "tail", "en")
    assert toolbox.lookup_abbreviation("z. B.", "de") == "zum Beispiel"
