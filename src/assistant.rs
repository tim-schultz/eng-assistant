use colored::*;
use mcp_client_rs::{Protocol, ToolResponseContent};
use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io::Read;
use std::process::Command;
use std::sync::{Arc, Mutex};

use anthropic_sdk::{AnthropicResponse, Client, ContentItem};
use anyhow::{anyhow, Context, Result};
use log::{debug, info, warn};
use serde_json::{json, Value};
use textwrap::{indent, wrap};

use crate::conversation_manager::{ConversationStorage, CurrentConversation, Message};
use crate::prompts::{BASE_SYSTEM_PROMPT, CHAIN_OF_THOUGHT_PROMPT};
use crate::tools::{ToolExecutor, TOOLS};
use crate::types::{MessageContent, ToolUseAssistant, ToolUseResult, ToolUseUser};
use dotenv::dotenv;

pub const MAX_CONTINUATION_ITERATIONS: i8 = 25;

pub struct Assistant<'a> {
    client: Client,
    system_prompt: String,
    conversation: CurrentConversation,
    tool_executor: ToolExecutor,
    protocols: Option<Vec<&'a Protocol>>,
}

/// A client for interacting with the Anthropic API with support for tools, streaming, and conversation management.
///
/// The `Assistant` struct provides a high-level interface for:
/// - Sending messages to Claude models
/// - Managing conversation history
/// - Executing tools/functions
/// - Processing streaming responses
/// - Handling protocol-based tools
///
/// # Examples
///
/// ```rust
/// use your_crate::Assistant;
///
/// #[tokio::main]
/// async fn main() -> Result<()> {
///     // Initialize a new Assistant
///     let mut assistant = Assistant::new(
///         "claude-3",
///         false,
///         None,
///         None,
///         None
///     )?;
///
///     // Send a message and get a response
///     let response = assistant.send_message("Hello!", false).await?;
///     println!("Response: {:?}", response);
///     
///     Ok(())
/// }
/// ```
///
/// # Tool Usage
///
/// The Assistant supports executing custom tools and protocols:
///
/// ```rust
/// use serde_json::json;
///
/// let custom_tools = json!([{
///     "name": "custom_tool",
///     "description": "A custom tool",
///     "parameters": {
///         "type": "object",
///         "properties": {
///             "param": {"type": "string"}
///         }
///     }
/// }]);
///
/// let mut assistant = Assistant::new(
///     "claude-3",
///     false,
///     Some(&custom_tools),
///     None,
///     None
/// )?;
/// ```
impl<'a> Assistant<'a> {
    /// Creates a new `Assistant` instance with the specified configuration.
    ///
    /// # Arguments
    ///
    /// * `model` - The Anthropic model to use (e.g., "claude-3")
    /// * `streaming` - Whether to enable streaming responses
    /// * `tools` - Optional custom tools configuration
    /// * `protocols` - Optional vector of protocols for specialized tool handling
    /// * `custom_system_prompt` - Optional custom system prompt
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the new `Assistant` instance or an error.
    ///
    /// # Errors
    ///
    /// Will return an error if:
    /// - The ANTHROPIC_API_KEY_RS environment variable is not set
    /// - Tool initialization fails
    /// - Client creation fails
    ///
    /// # Examples
    ///
    /// ```rust
    /// let assistant = Assistant::new(
    ///     "claude-3",
    ///     false,
    ///     None,
    ///     None,
    ///     None
    /// )?;
    /// ```
    pub fn new(
        model: &str,
        streaming: bool,
        tools: Option<&Value>,
        protocols: Option<Vec<&'a Protocol>>,
        custom_system_prompt: Option<String>,
    ) -> Result<Self> {
        dotenv().ok();

        let active_tools = Self::merge_tools(tools)?;
        let client = Self::create_client(model, streaming, &active_tools)?;
        let system_prompt = custom_system_prompt
            .unwrap_or_else(|| format!("\n{}\n{}", BASE_SYSTEM_PROMPT, CHAIN_OF_THOUGHT_PROMPT));

        let tool_client = client.clone().system(&system_prompt);
        let tool_executor = ToolExecutor::new(tool_client)?;

        Ok(Self {
            client,
            system_prompt,
            conversation: CurrentConversation::new(),
            tool_executor,
            protocols,
        })
    }

    /// Create an Anthropic client with specified configuration
    fn create_client(model: &str, streaming: bool, tools: &Value) -> Result<Client> {
        let api_key =
            std::env::var("ANTHROPIC_API_KEY_RS").context("Missing ANTHROPIC_API_KEY_RS")?;

        Ok(Client::new()
            .auth(&api_key)
            .model(model)
            .stream(streaming)
            .max_tokens(4000)
            .tools(tools)
            .beta("prompt-caching-2024-07-31"))
    }

    /// Merge custom tools with default tools
    fn merge_tools(custom_tools: Option<&Value>) -> Result<Value> {
        match custom_tools {
            Some(tools) => {
                let mut tool_map: HashMap<String, &Value> = HashMap::new();

                if let Some(tools_array) = tools.as_array() {
                    for tool in tools_array {
                        if let Some(name) = tool.get("name").and_then(|n| n.as_str()) {
                            tool_map.insert(name.to_string(), tool);
                        }
                    }
                }

                if let Some(default_tools) = TOOLS.as_array() {
                    for tool in default_tools {
                        if let Some(name) = tool.get("name").and_then(|n| n.as_str()) {
                            tool_map.insert(name.to_string(), tool);
                        }
                    }
                }

                Ok(json!(tool_map.values().cloned().collect::<Vec<_>>()))
            }
            None => Ok(TOOLS.clone()),
        }
    }

    /// Processes a message with optional tool choice configuration.
    ///
    /// # Arguments
    ///
    /// * `messages` - The messages to process
    /// * `tool_choice` - Optional tool choice configuration
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the `AnthropicResponse` or an error.
    async fn process_message(
        &mut self,
        messages: Value,
        tool_choice: Option<Value>,
    ) -> Result<AnthropicResponse> {
        let mut request = self
            .client
            .clone()
            .messages(&messages)
            .system(&self.system_prompt);

        if let Some(choice) = tool_choice {
            request = request.tool_choice(choice);
        }

        let request = request.build().context("Failed to build request")?;

        request
            .execute_and_return_json()
            .await
            .context("Failed to execute request")
    }

    /// Sends a message and receives a streaming response.
    ///
    /// # Arguments
    ///
    /// * `prompt` - The message to send
    /// * `clear_history` - Whether to clear conversation history before sending
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the response string or an error.
    pub async fn send_message_stream(
        &mut self,
        prompt: &str,
        clear_history: bool,
    ) -> Result<String> {
        let messages = self.prepare_conversation(prompt, clear_history)?;
        let message = Arc::new(Mutex::new(String::new()));
        let message_clone = message.clone();

        let request = self
            .client
            .clone()
            .messages(&messages)
            .system(&self.system_prompt)
            .build()?;

        request
            .execute(move |text| {
                let msg = message_clone.clone();
                async move {
                    println!("{text}");
                    if let Ok(mut locked_msg) = msg.lock() {
                        *locked_msg = format!("{}{}", *locked_msg, text);
                    }
                }
            })
            .await
            .context("Failed to execute streaming request")?;

        let final_message = {
            let locked = message.lock().unwrap();
            locked.clone()
        };

        Ok(final_message)
    }

    /// Executes a specific tool with the given parameters.
    ///
    /// # Arguments
    ///
    /// * `id` - The unique identifier for this tool execution
    /// * `name` - The name of the tool to execute
    /// * `input` - The input parameters for the tool
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the `ToolUseResult` or an error.
    async fn execute_tool(
        &mut self,
        id: String,
        name: String,
        input: Value,
    ) -> Result<ToolUseResult> {
        info!("Executing tool: {} (ID: {})", name, id);
        let tool_use_info = format!("Tool Use: {} ({}), Input: {:?}", name, id, input);
        let wrapped_info = wrap(&tool_use_info, 80);
        let formatted_info = indent(&wrapped_info.join("\n"), "    ");
        info!("{}", formatted_info);

        let result = match self.get_mcp_executor(&name).await {
            Ok(protocol) => {
                info!("Using MCP protocol for tool: {}", name);
                let response = protocol
                    .call_tool(&name, serde_json::to_value(&input)?)
                    .await
                    .with_context(|| format!("Failed to execute tool: {}", name))?;

                let mut result = String::new();
                for content in response.content {
                    match content {
                        ToolResponseContent::Text { text } => result.push_str(&text),
                        ToolResponseContent::Image { .. } => {
                            warn!("Ignoring image content from tool response")
                        }
                        ToolResponseContent::Resource { .. } => {
                            warn!("Ignoring resource content from tool response")
                        }
                    }
                }
                result
            }
            Err(e) => {
                debug!(
                    "MCP executor not found, falling back to default tool executor: {}",
                    e
                );
                self.tool_executor
                    .execute_tool(&name, &input)
                    .await
                    .with_context(|| format!("Failed to execute tool: {}", name))?
            }
        };

        info!("Tool execution completed: {}", name);
        Ok(ToolUseResult {
            id,
            name,
            input,
            tool_result: result,
        })
    }

    /// Processes tool responses from content items.
    ///
    /// # Arguments
    ///
    /// * `content` - Vector of content items to process
    ///
    /// # Returns
    ///
    /// Returns a tuple containing the response text and vector of tool results.
    pub async fn process_tools(
        &mut self,
        content: Vec<ContentItem>,
    ) -> Result<(String, Vec<ToolUseResult>)> {
        info!("Processing tool responses from content items");
        let mut response_text = String::new();
        let mut tool_results = Vec::new();

        for item in content {
            match item {
                ContentItem::Text { text } => {
                    debug!("Processing text content");
                    response_text.push_str(&text);
                    response_text.push('\n');
                }
                ContentItem::ToolUse { id, name, input } => {
                    info!("Processing tool use: {} ({})", name, id);
                    let result = self.execute_tool(id, name, input).await?;
                    tool_results.push(result);
                }
            }
        }

        info!("Processed {} tool results", tool_results.len());
        Ok((response_text, tool_results))
    }

    /// Initiates a chat session with tool support.
    ///
    /// # Arguments
    ///
    /// * `prompt` - The initial message to send
    /// * `clear_history` - Whether to clear conversation history
    /// * `max_iterations` - Optional maximum number of tool interaction iterations
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the final response string or an error.
    pub async fn chat_with_tools(
        &mut self,
        prompt: &str,
        clear_history: bool,
        max_iterations: Option<i8>,
    ) -> Result<String> {
        info!("Starting chat with tools session");
        let mut response_text = String::new();
        let mut response = self.send_message(prompt, clear_history).await?;
        let mut iterations = 0;
        let max_iter = max_iterations.unwrap_or(MAX_CONTINUATION_ITERATIONS);

        info!("Initial message sent, beginning tool processing loop");
        while iterations < max_iter {
            info!("Processing iteration {}/{}", iterations + 1, max_iter);
            let (partial_response, tool_results) = self.process_tools(response.content).await?;
            response_text.push_str(&partial_response);

            if tool_results.is_empty() {
                info!("No more tools to process, ending conversation");
                break;
            }

            debug!("Processing {} tool results", tool_results.len());
            response = self.process_tool_results(tool_results).await?;
            iterations += 1;
        }

        if iterations >= max_iter {
            warn!(
                "Maximum iterations ({}) reached, forcing conversation end",
                max_iter
            );
        }

        Ok(response_text)
    }

    // Updates to process_tool_results method:
    async fn process_tool_results(
        &mut self,
        tool_results: Vec<ToolUseResult>,
    ) -> Result<AnthropicResponse> {
        info!("Processing {} tool results", tool_results.len());
        for result in &tool_results {
            debug!(
                "Adding tool result to conversation: {} ({})",
                result.name, result.id
            );
            self.conversation.create(Message {
                role: "assistant".to_string(),
                content: MessageContent::ToolUseAssistant(vec![ToolUseAssistant {
                    tool_type: "tool_use".to_string(),
                    id: result.id.clone(),
                    name: result.name.clone(),
                    input: result.input.clone(),
                }]),
            });

            self.conversation.create(Message {
                role: "user".to_string(),
                content: MessageContent::ToolUseUser(vec![ToolUseUser {
                    tool_type: "tool_result".to_string(),
                    tool_use_id: result.id.clone(),
                    content: result.tool_result.clone(),
                }]),
            });
        }

        info!("Tool results added to conversation, processing response");
        let messages = self.get_messages_vec()?;
        let response = self.process_message(messages, None).await?;
        self.print_response(&response)?;

        Ok(response)
    }

    /// Sends a message and receives a response.
    ///
    /// # Arguments
    ///
    /// * `prompt` - The message to send
    /// * `clear_history` - Whether to clear conversation history
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the `AnthropicResponse` or an error.
    pub async fn send_message(
        &mut self,
        prompt: &str,
        clear_history: bool,
    ) -> Result<AnthropicResponse> {
        info!("Preparing to send message");
        let messages = self.prepare_conversation(prompt, clear_history)?;
        info!("Message prepared, sending to Anthropic API");
        let response = self.process_message(messages, None).await?;
        info!("Response received successfully");
        Ok(response)
    }

    /// Prepares the conversation by adding a new message and optionally clearing history.
    ///
    /// # Arguments
    ///
    /// * `prompt` - The message to add
    /// * `clear_history` - Whether to clear conversation history
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the prepared messages as a `Value` or an error.
    fn prepare_conversation(&mut self, prompt: &str, clear_history: bool) -> Result<Value> {
        if clear_history {
            info!("Clearing conversation history");
            self.conversation.delete_all();
        }

        info!("Adding new message to conversation");
        self.conversation.create(Message {
            role: "user".to_string(),
            content: MessageContent::Text(prompt.to_string()),
        });

        let messages = self.get_messages_vec()?;
        debug!(
            "Conversation prepared with {} messages",
            serde_json::from_value::<Vec<Message>>(messages.clone())?.len()
        );
        Ok(messages)
    }

    /// Gets the MCP executor for a specific tool.
    ///
    /// # Arguments
    ///
    /// * `tool_name` - The name of the tool
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing a reference to the protocol or an error.
    async fn get_mcp_executor(&mut self, tool_name: &str) -> Result<&Protocol> {
        if let Some(protocols) = &self.protocols {
            for protocol in protocols {
                let tools = protocol.list_tools().await?;
                if tools.tools.iter().any(|t| t.name == tool_name) {
                    return Ok(protocol);
                }
            }
        }
        Err(anyhow!("Tool not found in any protocol"))
    }

    /// Gets serialized messages from the conversation.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the serialized messages as a `Value` or an error.
    fn get_messages_vec(&self) -> Result<Value> {
        let conversation = self.conversation.read();
        serde_json::to_value(&conversation).context("Failed to serialize messages")
    }

    /// Print formatted response
    fn print_response(&self, response: &AnthropicResponse) -> Result<()> {
        println!("\n{}", "=".repeat(80).bright_blue());
        println!(
            "{}: {}",
            "Response ID".bright_yellow(),
            response.id.bright_white()
        );

        println!(
            "{}: {} | {}: {} | {}: {}",
            "Model".bright_yellow(),
            response.model.bright_white(),
            "Role".bright_yellow(),
            response.role.bright_white(),
            "Stop Reason".bright_yellow(),
            response.stop_reason.bright_white()
        );

        println!(
            "{}: {} | {}: {}",
            "Input Tokens".bright_yellow(),
            response.usage.input_tokens.to_string().bright_white(),
            "Output Tokens".bright_yellow(),
            response.usage.output_tokens.to_string().bright_white()
        );

        println!("{}", "=".repeat(80).bright_blue());

        for content in &response.content {
            match content {
                ContentItem::Text { text } => {
                    println!("\n{}", "TEXT CONTENT:".bright_green());
                    println!("{}", text);
                }
                ContentItem::ToolUse { id, name, input } => {
                    println!("\n{}", "TOOL USE:".bright_green());
                    println!("{}: {}", "Tool ID".bright_yellow(), id);
                    println!("{}: {}", "Tool Name".bright_yellow(), name);
                    println!("\n{}", "Tool Input:".bright_yellow());
                    println!("{}", serde_json::to_string_pretty(input)?);
                }
            }
        }

        println!("\n{}", "=".repeat(80).bright_blue());
        Ok(())
    }

    /// Loads a prompt from a file.
    ///
    /// # Arguments
    ///
    /// * `file_path` - Path to the prompt file
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the prompt string or an error.
    pub fn load_prompt(&self, file_path: &str) -> Result<String> {
        match File::open(file_path) {
            Ok(mut file) => {
                let mut contents = String::new();
                file.read_to_string(&mut contents)
                    .context("Failed to read file contents")?;
                Ok(contents)
            }
            Err(_) => Ok(String::new()),
        }
    }

    /// Opens a text editor with an optional template for prompt editing.
    ///
    /// # Arguments
    ///
    /// * `template` - Optional template content
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the edited prompt string or an error.
    pub fn edit_prompt(&self, template: Option<String>) -> Result<String> {
        let file_path = "prompt.txt";
        let existing_content = self.load_prompt(file_path)?;

        let initial_content = match template {
            Some(template_content) if !existing_content.contains(&template_content[..15]) => {
                template_content + &existing_content
            }
            _ => existing_content,
        };

        fs::write(file_path, &initial_content)?;

        for (editor, args) in [("code", vec!["--wait"]), ("vim", vec![]), ("nano", vec![])] {
            match Command::new(editor).arg(file_path).args(args).status() {
                Ok(status) => {
                    info!("{} editor exited with status: {}", editor, status);
                    return self.load_prompt(file_path);
                }
                Err(e) => warn!("Failed to open {} editor: {}", editor, e),
            }
        }

        warn!("No suitable editor found");
        self.load_prompt(file_path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mockall::predicate::*;
    use serde_json::json;
    use std::env;
    use tokio;
    const TEST_API_KEY: &str = "test_api_key_12345";

    // Helper function to set up environment for tests
    fn setup_test_env() {
        env::set_var("ANTHROPIC_API_KEY_RS", TEST_API_KEY);
    }

    // Helper function to clean up environment after tests
    fn cleanup_test_env() {
        env::remove_var("ANTHROPIC_API_KEY_RS");
    }

    // Helper function to create a test assistant
    async fn setup_test_assistant() -> Result<Assistant<'static>> {
        setup_test_env();

        let test_tools = json!([
            {
                "name": "test_tool",
                "description": "A test tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "param": {"type": "string"}
                    },
                    "required": ["param"]
                }
            }
        ]);

        Assistant::new("claude-3", false, Some(&test_tools), None, None)
    }

    #[test]
    fn test_assistant_initialization() {
        setup_test_env();
        let assistant = Assistant::new("claude-3", false, None, None, None);
        assert!(assistant.is_ok());

        let assistant_with_tools = Assistant::new(
            "claude-3",
            false,
            Some(&json!([{"name": "custom_tool", "description": "A custom tool"}])),
            None,
            None,
        );
        assert!(assistant_with_tools.is_ok());
        cleanup_test_env();
    }

    #[test]
    fn test_merge_tools() {
        setup_test_env();
        // Test merging custom tools with default tools
        let custom_tools = json!([{
            "name": "custom_tool",
            "description": "A custom tool"
        }]);

        let merged = Assistant::merge_tools(Some(&custom_tools));
        assert!(merged.is_ok());

        let tools = merged.unwrap();
        assert!(tools.as_array().unwrap().iter().any(|tool| tool
            .get("name")
            .unwrap()
            .as_str()
            .unwrap()
            == "custom_tool"));
        cleanup_test_env();
    }

    #[tokio::test]
    async fn test_process_message() -> Result<()> {
        let mut assistant = setup_test_assistant().await?;

        let messages = json!([{
            "role": "user",
            "content": "test message"
        }]);

        let response = assistant.process_message(messages, None).await;
        assert!(response.is_ok());
        cleanup_test_env();
        Ok(())
    }

    #[tokio::test]
    async fn test_execute_tool() -> Result<()> {
        let mut assistant = setup_test_assistant().await?;

        let result = assistant
            .execute_tool(
                "test_id".to_string(),
                "test_tool".to_string(),
                json!({"param": "test_value"}),
            )
            .await;

        assert!(result.is_ok());
        let tool_result = result.unwrap();
        assert_eq!(tool_result.id, "test_id");
        assert_eq!(tool_result.name, "test_tool");
        cleanup_test_env();
        Ok(())
    }

    #[tokio::test]
    async fn test_chat_with_tools() -> Result<()> {
        let mut assistant = setup_test_assistant().await?;

        let response = assistant
            .chat_with_tools("Test prompt", true, Some(1))
            .await;

        assert!(response.is_ok());
        cleanup_test_env();
        Ok(())
    }

    #[test]
    fn test_prepare_conversation() -> Result<()> {
        setup_test_env();
        let mut assistant = Assistant::new("claude-3", false, None, None, None)?;

        let messages = assistant.prepare_conversation("test prompt", false)?;
        assert!(messages.is_array());

        let messages_vec = messages.as_array().unwrap();
        assert!(!messages_vec.is_empty());

        // Test clear history
        let cleared_messages = assistant.prepare_conversation("test prompt", true)?;
        let cleared_vec = cleared_messages.as_array().unwrap();
        assert_eq!(cleared_vec.len(), 1);

        cleanup_test_env();
        Ok(())
    }

    #[test]
    fn test_load_prompt() -> Result<()> {
        setup_test_env();
        let assistant = Assistant::new("claude-3", false, None, None, None)?;

        // Test non-existent file
        let result = assistant.load_prompt("nonexistent.txt");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "");

        cleanup_test_env();
        Ok(())
    }

    #[tokio::test]
    async fn test_process_tool_results() -> Result<()> {
        let mut assistant = setup_test_assistant().await?;

        let tool_results = vec![ToolUseResult {
            id: "test_id".to_string(),
            name: "test_tool".to_string(),
            input: json!({"param": "test_value"}),
            tool_result: "test result".to_string(),
        }];

        let response = assistant.process_tool_results(tool_results).await;
        assert!(response.is_ok());
        cleanup_test_env();
        Ok(())
    }
}
