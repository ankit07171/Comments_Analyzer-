"""
jira_client.py
----------------
Thin wrapper around the Jira Cloud REST API (v3) used to auto-file tickets
for high-priority / negative comments so a human owns follow-up instead of
the insight getting lost in a dashboard.

Deliberately does NOT use Zendesk, Slack, SendGrid, or HubSpot — this
project routes everything through Jira (ticketing) and the in-app alert
panel, per project scope.
"""

from __future__ import annotations

import os
import requests


class JiraClient:
    def __init__(self):
        self.domain = os.getenv("JIRA_DOMAIN", "")
        self.email = os.getenv("JIRA_EMAIL", "")
        self.token = os.getenv("JIRA_API_TOKEN", "")
        self.project_key = os.getenv("JIRA_PROJECT_KEY", "CS")
        self.issue_type = os.getenv("JIRA_ISSUE_TYPE", "Task")

    def is_configured(self) -> bool:
        return bool(self.domain and self.email and self.token)

    def _base_url(self) -> str:
        domain = self.domain
        if not domain.startswith("http"):
            domain = f"https://{domain}"
        return domain.rstrip("/")

    def create_issue(self, summary: str, description: str, priority: str = "High",
                      labels: list | None = None) -> dict:
        """Creates a Jira issue. Returns {"ok": bool, "key": str|None, "url": str|None, "error": str|None}."""
        if not self.is_configured():
            return {"ok": False, "key": None, "url": None, "error": "Jira not configured"}

        labels = labels or ["comment-intelligence", "auto-flagged"]

        payload = {
            "fields": {
                "project": {"key": self.project_key},
                "summary": summary[:250],
                "description": {
                    "type": "doc",
                    "version": 1,
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [{"type": "text", "text": description[:1900]}],
                        }
                    ],
                },
                "issuetype": {"name": self.issue_type},
                "labels": labels,
            }
        }

        try:
            resp = requests.post(
                f"{self._base_url()}/rest/api/3/issue",
                json=payload,
                auth=(self.email, self.token),
                headers={"Content-Type": "application/json"},
                timeout=15,
            )
            if resp.status_code in (200, 201):
                data = resp.json()
                key = data.get("key")
                url = f"{self._base_url()}/browse/{key}" if key else None
                return {"ok": True, "key": key, "url": url, "error": None}
            return {"ok": False, "key": None, "url": None, "error": f"{resp.status_code}: {resp.text[:200]}"}
        except Exception as e:
            return {"ok": False, "key": None, "url": None, "error": str(e)}


_client_singleton: JiraClient | None = None


def get_client() -> JiraClient:
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = JiraClient()
    return _client_singleton
