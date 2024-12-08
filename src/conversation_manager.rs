use crate::types::MessageContent;
use chrono::Local;
use log::{info, warn};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs::File;
use std::io::Write;

// Add Sized bound explicitly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: MessageContent,
}

pub trait ConversationStorage {
    fn create(&mut self, message: Message);
    fn read(&self) -> Vec<Message>;
    fn update(&mut self, index: usize, message: Message) -> Option<Message>;
    fn delete(&mut self, index: usize) -> Option<Message>;
    fn delete_all(&mut self);
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Debug, Clone)]
pub struct CurrentConversation {
    messages: Vec<Message>,
}

impl CurrentConversation {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
        }
    }
}

impl ConversationStorage for CurrentConversation {
    fn create(&mut self, message: Message) {
        info!(
            "Creating message in current conversation starting with - {}",
            message.role
        );
        self.messages.push(message);
    }

    fn read(&self) -> Vec<Message> {
        self.messages.clone()
    }

    fn update(&mut self, index: usize, message: Message) -> Option<Message> {
        if index < self.messages.len() {
            info!(
                "Updating message in current conversation at index {}",
                index
            );
            Some(std::mem::replace(&mut self.messages[index], message))
        } else {
            None
        }
    }

    fn delete(&mut self, index: usize) -> Option<Message> {
        if index < self.messages.len() {
            info!(
                "Deleting message from current conversation at index {}",
                index
            );
            Some(self.messages.remove(index))
        } else {
            None
        }
    }

    fn delete_all(&mut self) {
        info!("Deleting all messages from current conversation");
        self.messages.clear();
    }

    fn len(&self) -> usize {
        self.messages.len()
    }
}

#[derive(Debug, Clone)]
pub struct ConversationHistory {
    messages: VecDeque<Message>,
    max_size: usize,
}

impl ConversationHistory {
    pub fn new(max_size: usize) -> Self {
        Self {
            messages: VecDeque::new(),
            max_size,
        }
    }
}

impl ConversationStorage for ConversationHistory {
    fn create(&mut self, message: Message) {
        if self.messages.len() >= self.max_size {
            self.messages.pop_front();
        }
        info!("Creating message in history");
        self.messages.push_back(message);
    }

    fn read(&self) -> Vec<Message> {
        Vec::from(self.messages.clone())
    }

    fn update(&mut self, index: usize, message: Message) -> Option<Message> {
        if index < self.messages.len() {
            info!("Updating message in history at index {}", index);
            Some(std::mem::replace(&mut self.messages[index], message))
        } else {
            None
        }
    }

    fn delete(&mut self, index: usize) -> Option<Message> {
        if index < self.messages.len() {
            info!("Deleting message from history at index {}", index);
            Some(self.messages.remove(index).unwrap())
        } else {
            None
        }
    }

    fn delete_all(&mut self) {
        info!("Deleting all messages from history");
        self.messages.clear();
    }

    fn len(&self) -> usize {
        self.messages.len()
    }
}

#[derive(Debug)]
pub struct ConversationManager {
    current: CurrentConversation,
    history: ConversationHistory,
}

impl ConversationManager {
    pub fn new(max_history_size: usize) -> Self {
        info!("Creating new ConversationManager");
        Self {
            current: CurrentConversation::new(),
            history: ConversationHistory::new(max_history_size),
        }
    }

    pub fn current(&mut self) -> &mut CurrentConversation {
        &mut self.current
    }

    pub fn history(&mut self) -> &mut ConversationHistory {
        &mut self.history
    }

    pub fn move_current_to_history(&mut self) {
        info!("Moving current conversation to history");
        let messages = self.current.read();
        for message in messages {
            self.history.create(message);
        }
        self.current.delete_all();
    }

    pub fn read_all(&self) -> Vec<Message> {
        let mut all = self.history.read();
        all.extend(self.current.read());
        all
    }

    pub fn save_to_file(&self) -> std::io::Result<String> {
        let filename = format!("Chat_{}.md", Local::now().format("%H%M"));
        let mut content = String::from("# Chat Log\n\n");

        for message in self.read_all() {
            match message.role.as_str() {
                "user" => {
                    content.push_str("## User\n\n");
                    if let MessageContent::Text(text) = message.content {
                        content.push_str(&format!("{}\n\n", text));
                    }
                }
                "assistant" => {
                    content.push_str("## Assistant\n\n");
                    if let MessageContent::Text(text) = message.content {
                        content.push_str(&format!("{}\n\n", text));
                    }
                }
                _ => warn!("Unknown message role: {}", message.role),
            }
        }

        let mut file = File::create(&filename)?;
        file.write_all(content.as_bytes())?;
        Ok(filename)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_message(role: &str, content: &str) -> Message {
        Message {
            role: role.to_string(),
            content: MessageContent::Text(content.to_string()),
        }
    }

    #[test]
    fn test_conversation_storage() {
        let mut manager = ConversationManager::new(5);

        let msg = create_test_message("user", "Hello");
        manager.current().create(msg.clone());
        assert_eq!(manager.current().len(), 1);

        manager.history().create(msg.clone());
        assert_eq!(manager.history().len(), 1);

        let new_msg = create_test_message("user", "Updated");
        assert!(manager.current().update(0, new_msg.clone()).is_some());
        assert!(manager.history().update(0, new_msg).is_some());

        assert!(manager.current().delete(0).is_some());
        assert!(manager.history().delete(0).is_some());

        assert!(manager.current().is_empty());
        assert!(manager.history().is_empty());
    }

    #[test]
    fn test_move_current_to_history() {
        let mut manager = ConversationManager::new(5);

        manager
            .current()
            .create(create_test_message("user", "Current"));
        assert_eq!(manager.current().len(), 1);
        assert_eq!(manager.history().len(), 0);

        manager.move_current_to_history();
        assert_eq!(manager.current().len(), 0);
        assert_eq!(manager.history().len(), 1);
    }
}
