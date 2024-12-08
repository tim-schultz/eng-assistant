use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    ToolUseAssistant(Vec<ToolUseAssistant>),
    ToolUseUser(Vec<ToolUseUser>),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ToolUseAssistant {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub id: String,
    pub name: String,
    pub input: Value,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ToolUseUser {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub tool_use_id: String,
    pub content: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ToolUseResult {
    pub id: String,
    pub name: String,
    pub input: Value,
    pub tool_result: String,
}
