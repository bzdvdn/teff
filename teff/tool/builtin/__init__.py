from teff.memory.tool import MemoryTool
from teff.tool.builtin.calculator import CalculatorTool
from teff.tool.builtin.csv import CsvQueryTool
from teff.tool.builtin.data import (
    JsonParseTool,
    KVStoreTool,
    PythonEvalTool,
    YamlParseTool,
)
from teff.tool.builtin.file import EditFileTool, ReadFileTool, WriteFileTool
from teff.tool.builtin.fs import CurrentTimeTool, GetEnvTool, GlobTool, ListDirTool
from teff.tool.builtin.git import GitTool
from teff.tool.builtin.github import (
    GitHubApproveTool,
    GitHubGetPRChangesTool,
    GitHubListOpenPRsTool,
    GitHubPostCommentTool,
)
from teff.tool.builtin.gitlab import (
    GitLabApproveTool,
    GitLabGetMRChangesTool,
    GitLabListOpenMRsTool,
    GitLabPostNoteTool,
)
from teff.tool.builtin.http import HttpRequestTool
from teff.tool.builtin.human import AskHuman
from teff.tool.builtin.reply import ReplyToUser
from teff.tool.builtin.lock import LockTool
from teff.tool.builtin.notify import SendEmailTool, SendTelegramTool
from teff.tool.builtin.pdf import PDFReadTool
from teff.tool.builtin.rag_ingest import RAGIngestTool
from teff.tool.builtin.redis import RedisTool
from teff.tool.builtin.s3 import S3GetTool, S3PutTool, S3Tool
from teff.tool.builtin.shell import ShellTool
from teff.tool.builtin.slack import SlackSendTool
from teff.tool.builtin.sql import SQLDescribeTool, SQLListTablesTool, SQLQueryTool
from teff.tool.builtin.wait_for import WaitForTool
from teff.tool.builtin.web_fetch import WebFetchTool
from teff.tool.builtin.web_search import WebSearchTool
from teff.tool.registry import default_tool_registry

default_tool_registry.register(CalculatorTool)
default_tool_registry.register(ShellTool)
default_tool_registry.register(ReadFileTool)
default_tool_registry.register(WriteFileTool)
default_tool_registry.register(EditFileTool)
default_tool_registry.register(WebSearchTool)
default_tool_registry.register(WebFetchTool)
default_tool_registry.register(PDFReadTool)
default_tool_registry.register(S3Tool)
default_tool_registry.register(S3GetTool)
default_tool_registry.register(S3PutTool)
default_tool_registry.register(SlackSendTool)
default_tool_registry.register(SQLQueryTool)
default_tool_registry.register(SQLListTablesTool)
default_tool_registry.register(SQLDescribeTool)
default_tool_registry.register(ListDirTool)
default_tool_registry.register(GlobTool)
default_tool_registry.register(GetEnvTool)
default_tool_registry.register(CurrentTimeTool)
default_tool_registry.register(JsonParseTool)
default_tool_registry.register(YamlParseTool)
default_tool_registry.register(KVStoreTool)
default_tool_registry.register(PythonEvalTool)
default_tool_registry.register(HttpRequestTool)
default_tool_registry.register(SendEmailTool)
default_tool_registry.register(SendTelegramTool)
default_tool_registry.register(RedisTool)
default_tool_registry.register(GitTool)
default_tool_registry.register(LockTool)
default_tool_registry.register(WaitForTool)
default_tool_registry.register(CsvQueryTool)
default_tool_registry.register(GitLabListOpenMRsTool)
default_tool_registry.register(GitLabGetMRChangesTool)
default_tool_registry.register(GitLabPostNoteTool)
default_tool_registry.register(GitLabApproveTool)
default_tool_registry.register(GitHubListOpenPRsTool)
default_tool_registry.register(GitHubGetPRChangesTool)
default_tool_registry.register(GitHubPostCommentTool)
default_tool_registry.register(GitHubApproveTool)
default_tool_registry.register(MemoryTool)
default_tool_registry.register(AskHuman)
default_tool_registry.register(ReplyToUser)
default_tool_registry.register(RAGIngestTool)

__all__ = [
    "CalculatorTool",
    "ShellTool",
    "ReadFileTool",
    "WriteFileTool",
    "EditFileTool",
    "WebSearchTool",
    "WebFetchTool",
    "PDFReadTool",
    "S3Tool",
    "S3GetTool",
    "S3PutTool",
    "SlackSendTool",
    "SQLQueryTool",
    "SQLListTablesTool",
    "SQLDescribeTool",
    "ListDirTool",
    "GlobTool",
    "GetEnvTool",
    "CurrentTimeTool",
    "JsonParseTool",
    "YamlParseTool",
    "KVStoreTool",
    "PythonEvalTool",
    "HttpRequestTool",
    "SendEmailTool",
    "SendTelegramTool",
    "RedisTool",
    "GitTool",
    "LockTool",
    "WaitForTool",
    "CsvQueryTool",
    "GitLabListOpenMRsTool",
    "GitLabGetMRChangesTool",
    "GitLabPostNoteTool",
    "GitLabApproveTool",
    "GitHubListOpenPRsTool",
    "GitHubGetPRChangesTool",
    "GitHubPostCommentTool",
    "GitHubApproveTool",
    "MemoryTool",
    "AskHuman",
    "ReplyToUser",
    "RAGIngestTool",
]
