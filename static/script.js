class ResearchAgentUI {
    constructor() {
        this.websocket = null;
        this.sessionId = null;
        this.currentModel = 'ollama';
        this.currentMessages = [];
        this.isProcessing = false;
        this.accessKey = null;
        this.isAccessVerified = false;
        
        this.initializeElements();
        this.attachEventListeners();
        this.setupMessageInput();
        this.updateUIState();
    }
    
    initializeElements() {
        // Main UI elements
        this.settingsDropdown = document.getElementById('settingsDropdown');
        this.settingsBtn = document.getElementById('settingsBtn');
        this.modelSelect = document.getElementById('modelSelect');
        this.testConnectionBtn = document.getElementById('testConnectionBtn');
        
        // Access key elements
        this.accessKeyInput = document.getElementById('accessKeyInput');
        this.verifyKeyBtn = document.getElementById('verifyKeyBtn');
        this.accessStatus = document.getElementById('accessStatus');
        this.accessIcon = document.getElementById('accessIcon');
        this.accessText = document.getElementById('accessText');
        
        // Chat elements
        this.welcomeScreen = document.getElementById('welcomeScreen');
        this.chatMessages = document.getElementById('chatMessages');
        this.messageInput = document.getElementById('messageInput');
        this.sendBtn = document.getElementById('sendBtn');
        
        // Status elements
        this.connectionIconNav = document.getElementById('connectionIconNav');
        this.connectionTextNav = document.getElementById('connectionTextNav');
        this.statusDisplay = document.getElementById('statusDisplay');
        this.statusMessage = document.getElementById('statusMessage');
        this.statusDetails = document.getElementById('statusDetails');
        this.progressBar = document.getElementById('progressBar');
        
        // Overlays
        this.loadingOverlay = document.getElementById('loadingOverlay');
        this.errorModal = document.getElementById('errorModal');
        this.errorMessage = document.getElementById('errorMessage');
        this.closeErrorModal = document.getElementById('closeErrorModal');
        this.retryBtn = document.getElementById('retryBtn');
    }
    
    attachEventListeners() {
        // Settings dropdown
        this.settingsBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            this.toggleSettings();
        });
        this.modelSelect.addEventListener('change', (e) => this.changeModel(e.target.value));
        this.testConnectionBtn.addEventListener('click', () => this.testConnection());
        
        // Access key management
        this.verifyKeyBtn.addEventListener('click', () => this.verifyAccessKey());
        this.accessKeyInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                this.verifyAccessKey();
            }
        });
        
        // Chat input
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Error handling
        this.closeErrorModal.addEventListener('click', () => this.hideError());
        this.retryBtn.addEventListener('click', () => this.retryLastAction());
        
        // Click outside dropdown to close
        document.addEventListener('click', (e) => {
            if (!this.settingsDropdown.contains(e.target) && !this.settingsBtn.contains(e.target)) {
                this.closeSettings();
            }
        });
    }
    
    setupMessageInput() {
        // Auto-resize textarea
        this.messageInput.addEventListener('input', (e) => {
            e.target.style.height = 'auto';
            e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px';
            
            // Enable/disable send button
            const hasContent = e.target.value.trim().length > 0;
            this.sendBtn.disabled = !hasContent || this.isProcessing || !this.isAccessVerified;
        });
    }
    
    // Access key management
    async verifyAccessKey() {
        const accessKey = this.accessKeyInput.value.trim();
        if (!accessKey) {
            this.updateAccessStatus('Access key cannot be empty', false);
            return;
        }
        
        this.verifyKeyBtn.disabled = true;
        this.updateAccessStatus('Verifying...', false);
        
        try {
            // Test API call with access key to verify it's valid
            const response = await fetch('/health?model_type=' + this.currentModel, {
                headers: {
                    'Authorization': `Bearer ${accessKey}`
                }
            });
            
            if (response.ok) {
                this.accessKey = accessKey;
                this.isAccessVerified = true;
                this.updateAccessStatus('Access verified', true);
                this.updateUIState();
                this.checkConnection();
            } else if (response.status === 401) {
                this.updateAccessStatus('Invalid access key', false);
                this.isAccessVerified = false;
            } else {
                this.updateAccessStatus('Verification failed', false);
                this.isAccessVerified = false;
            }
        } catch (error) {
            this.updateAccessStatus('Connection failed', false);
            this.isAccessVerified = false;
        } finally {
            this.verifyKeyBtn.disabled = false;
            this.updateUIState();
        }
    }
    
    updateAccessStatus(message, verified) {
        this.accessText.textContent = message;
        this.accessStatus.className = 'access-status' + (verified ? ' verified' : (message.includes('failed') || message.includes('Invalid') ? ' error' : ''));
        
        if (verified) {
            this.accessIcon.className = 'fas fa-unlock';
        } else {
            this.accessIcon.className = 'fas fa-lock';
        }
    }
    
    updateUIState() {
        // Update form elements based on access verification
        this.modelSelect.disabled = !this.isAccessVerified;
        this.testConnectionBtn.disabled = !this.isAccessVerified;
        this.messageInput.disabled = !this.isAccessVerified;
        
        if (this.isAccessVerified) {
            this.messageInput.placeholder = "Describe your research topic or question...";
        } else {
            this.messageInput.placeholder = "Enter access key in settings first...";
            this.connectionTextNav.textContent = 'Access key required';
        }
        
        // Update send button state
        const hasContent = this.messageInput.value.trim().length > 0;
        this.sendBtn.disabled = !hasContent || this.isProcessing || !this.isAccessVerified;
    }
    
    // Settings management
    toggleSettings() {
        this.settingsDropdown.classList.toggle('show');
    }
    
    closeSettings() {
        this.settingsDropdown.classList.remove('show');
    }
    
    changeModel(model) {
        this.currentModel = model;
        if (this.isAccessVerified) {
            this.checkConnection();
        }
    }
    
    // Connection management
    async checkConnection() {
        if (!this.isAccessVerified) {
            this.updateConnectionStatus('Access key required', false);
            return;
        }
        
        this.updateConnectionStatus('Checking...', false);
        this.testConnectionBtn.disabled = true;
        
        try {
            const response = await fetch(`/health?model_type=${this.currentModel}`, {
                headers: {
                    'Authorization': `Bearer ${this.accessKey}`
                }
            });
            
            if (response.ok) {
                const data = await response.json();
                if (data.healthy) {
                    this.updateConnectionStatus('Connected', true);
                } else {
                    this.updateConnectionStatus('Service Unavailable', false);
                }
            } else {
                this.updateConnectionStatus('Connection Failed', false);
            }
        } catch (error) {
            this.updateConnectionStatus('Connection Failed', false);
        } finally {
            this.testConnectionBtn.disabled = false;
        }
    }
    
    async testConnection() {
        await this.checkConnection();
    }
    
    updateConnectionStatus(status, connected) {
        this.connectionTextNav.textContent = status;
        const indicator = this.connectionIconNav.parentElement;
        
        if (connected) {
            indicator.classList.add('connected');
        } else {
            indicator.classList.remove('connected');
        }
    }
    
    // Message handling
    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message || this.isProcessing || !this.isAccessVerified) return;
        
        // Hide welcome screen
        if (this.welcomeScreen) {
            this.welcomeScreen.style.display = 'none';
        }
        
        // Add user message
        this.addMessage(message, 'user');
        this.messageInput.value = '';
        this.messageInput.style.height = 'auto';
        this.sendBtn.disabled = true;
        this.isProcessing = true;
        
        // Show typing indicator
        this.addTypingIndicator();
        
        try {
            await this.processResearchRequest(message);
        } catch (error) {
            this.hideStatus();
            this.removeTypingIndicator();
            this.showError('Failed to process your request. Please try again.');
            console.error('Error processing request:', error);
        } finally {
            this.isProcessing = false;
            this.updateUIState();
        }
    }
    
    addMessage(content, type) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        
        // Create avatar
        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'message-avatar';
        
        if (type === 'user') {
            avatarDiv.innerHTML = '<i class="fas fa-user"></i>';
        } else {
            avatarDiv.innerHTML = '<i class="fas fa-robot"></i>';
        }
        
        // Create message wrapper
        const wrapperDiv = document.createElement('div');
        wrapperDiv.className = 'message-wrapper';
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        if (type === 'assistant' && content.includes('# ')) {
            messageContent.innerHTML = this.markdownToHtml(content);
        } else {
            messageContent.textContent = content;
        }
        
        const messageInfo = document.createElement('div');
        messageInfo.className = 'message-info';
        messageInfo.textContent = new Date().toLocaleTimeString();
        
        wrapperDiv.appendChild(messageContent);
        wrapperDiv.appendChild(messageInfo);
        
        messageDiv.appendChild(avatarDiv);
        messageDiv.appendChild(wrapperDiv);
        
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
        
        this.currentMessages.push({ content, type, timestamp: new Date() });
    }
    
    addTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'typing-indicator';
        typingDiv.id = 'typingIndicator';
        
        const dotsDiv = document.createElement('div');
        dotsDiv.className = 'typing-dots';
        
        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('div');
            dot.className = 'typing-dot';
            dotsDiv.appendChild(dot);
        }
        
        typingDiv.appendChild(dotsDiv);
        this.chatMessages.appendChild(typingDiv);
        this.scrollToBottom();
    }
    
    removeTypingIndicator() {
        const typingIndicator = document.getElementById('typingIndicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }
    
    scrollToBottom() {
        this.chatMessages.scrollTo({
            top: this.chatMessages.scrollHeight,
            behavior: 'smooth'
        });
    }
    
    // Process research request
    async processResearchRequest(topic) {
        try {
            // Create session
            const sessionResponse = await fetch('/research/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.accessKey}`
                },
                body: JSON.stringify({
                    topic: topic,
                    model_type: this.currentModel
                })
            });
            
            if (!sessionResponse.ok) {
                const errorData = await sessionResponse.json();
                throw new Error(errorData.message || 'Failed to start research session');
            }
            
            const sessionData = await sessionResponse.json();
            this.sessionId = sessionData.session_id;
            
            // Connect to WebSocket for real-time updates
            await this.connectWebSocket();
            
        } catch (error) {
            throw new Error(`Failed to process research request: ${error.message}`);
        }
    }
    
    async connectWebSocket() {
        const wsUrl = `ws://${window.location.host}/ws/${this.sessionId}?access_key=${this.accessKey}`;
        this.websocket = new WebSocket(wsUrl);
        
        this.websocket.onopen = () => {
            console.log('WebSocket connected');
        };
        
        this.websocket.onmessage = (event) => {
            this.handleWebSocketMessage(JSON.parse(event.data));
        };
        
        this.websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.removeTypingIndicator();
            this.showError('Connection lost. Please try again.');
        };
        
        this.websocket.onclose = () => {
            console.log('WebSocket disconnected');
            this.websocket = null;
        };
    }
    
    handleWebSocketMessage(data) {
        console.log('Received WebSocket message:', data);
        
        switch(data.type) {
            case 'status':
                this.showStatus(data.message, '');
                break;
            case 'step':
                this.updateAgentStep(data);
                break;
            case 'completion':
                this.hideStatus();
                this.removeTypingIndicator();
                this.showFinalResult(data.final_result);
                break;
            case 'error':
                this.hideStatus();
                this.removeTypingIndicator();
                this.showError(data.message);
                break;
            case 'connection':
                console.log('Connected:', data.message);
                break;
        }
    }
    
    showStatus(message, details = '') {
        this.statusDisplay.style.display = 'block';
        this.statusMessage.textContent = message;
        this.statusDetails.textContent = details;
        this.scrollToBottom();
    }
    
    hideStatus() {
        this.statusDisplay.style.display = 'none';
    }
    
    updateAgentStep(data) {
        const agentNames = {
            'planner': '📋 Planner',
            'researcher': '🔍 Researcher',
            'synthesizer': '📝 Creation',
            'critic': '🔍 Reviewer',
            'reporter': '📊 Reporter'
        };
        
        const agentName = agentNames[data.node_name] || data.node_name;
        let message = `${agentName} is working...`;
        let details = '';
        
        if (data.details) {
            if (data.details.action_type === 'search') {
                details = `Search query: ${data.details.query || ''}`;
            } else if (data.details.action_type === 'synthesize') {
                details = 'Synthesizing research results...';
            }
            
            if (data.details.has_enough_content !== undefined) {
                details += data.details.has_enough_content ? ' (Content sufficient)' : ' (More content needed)';
            }
        }
        
        if (data.resource_count) {
            details += `\nFound ${data.resource_count} resources`;
        }
        
        if (data.research_gap) {
            details += `\nResearch gap: ${data.research_gap}`;
        }
        
        if (data.critic_result) {
            if (data.critic_result.is_valid) {
                details += '\n✅ Review passed';
            } else {
                details += '\n❌ Needs improvement';
                if (data.critic_result.feedback) {
                    details += `\nFeedback: ${data.critic_result.feedback}`;
                }
            }
        }
        
        this.showStatus(message, details);
        
        // Update progress (rough estimation based on step number)
        const progressPercent = Math.min((data.step_number / 10) * 100, 90);
        this.progressBar.style.width = `${progressPercent}%`;
    }
    
    updateProcessStep(step, status, content) {
        // Update or add process steps in the message
        let processSteps = document.querySelector('.process-steps');
        if (!processSteps) {
            processSteps = document.createElement('div');
            processSteps.className = 'process-steps';
            
            const typingIndicator = document.getElementById('typingIndicator');
            if (typingIndicator) {
                this.chatMessages.insertBefore(processSteps, typingIndicator);
            } else {
                this.chatMessages.appendChild(processSteps);
            }
        }
        
        // Find or create step element
        let stepElement = document.getElementById(`step-${step}`);
        if (!stepElement) {
            stepElement = document.createElement('div');
            stepElement.className = 'process-step';
            stepElement.id = `step-${step}`;
            stepElement.innerHTML = `
                <i class="fas fa-circle"></i>
                <span>${this.getStepDisplayName(step)}: ${status}</span>
            `;
            processSteps.appendChild(stepElement);
        } else {
            stepElement.querySelector('span').textContent = `${this.getStepDisplayName(step)}: ${status}`;
        }
        
        // Update step status styling
        stepElement.className = `process-step ${status === 'completed' ? 'completed' : 'active'}`;
        
        this.scrollToBottom();
    }
    
    getStepDisplayName(step) {
        const stepNames = {
            'planner': 'Planning',
            'researcher': 'Research',
            'synthesizer': 'Synthesis',
            'critic': 'Review',
            'reporter': 'Writing'
        };
        return stepNames[step] || step;
    }
    
    showFinalResult(finalResult) {
        // Progress bar to 100%
        this.progressBar.style.width = '100%';
        
        let content = '';
        
        if (finalResult.success && finalResult.proposal) {
            const proposal = finalResult.proposal;
            
            // Format references as a proper numbered list with links
            let referencesContent = '';
            if (proposal.references && Array.isArray(proposal.references)) {
                referencesContent = proposal.references
                    .map((ref, index) => {
                        const title = ref.title || 'Untitled';
                        const uri = ref.uri || '#';
                        const referenceNumber = index + 1;
                        return `${referenceNumber}. [${title}](${uri})`;
                    })
                    .join('\n');
            } else {
                referencesContent = 'No references available.';
            }
            
            content = `# ${proposal.title || 'Research Proposal'}

## Introduction
${proposal.introduction || ''}

<!-- -->

## Research Questions
${proposal.research_problem || ''}

<!-- -->

## Research Methods
${proposal.methodology || ''}

<!-- -->

## Expected Outcomes
${proposal.expected_outcomes || ''}

<!-- -->

## References

${referencesContent}`;
        } else {
            content = finalResult.message || 'Research process completed, but no final proposal was generated.';
        }
        
        // Create result message with avatar
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant';
        
        // Create avatar
        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'message-avatar';
        avatarDiv.innerHTML = '<i class="fas fa-robot"></i>';
        
        // Create message wrapper
        const wrapperDiv = document.createElement('div');
        wrapperDiv.className = 'message-wrapper';
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        // Create result container
        const resultContainer = document.createElement('div');
        resultContainer.className = 'result-content';
        resultContainer.innerHTML = this.markdownToHtml(content);
        
        // Add download button
        if (finalResult.success) {
            const downloadBtn = document.createElement('button');
            downloadBtn.className = 'download-btn';
            downloadBtn.innerHTML = '<i class="fas fa-download"></i> Download Proposal';
            downloadBtn.addEventListener('click', () => this.downloadResult(content));
            resultContainer.appendChild(downloadBtn);
        }
        
        messageContent.appendChild(resultContainer);
        
        const messageInfo = document.createElement('div');
        messageInfo.className = 'message-info';
        messageInfo.textContent = new Date().toLocaleTimeString();
        
        wrapperDiv.appendChild(messageContent);
        wrapperDiv.appendChild(messageInfo);
        
        messageDiv.appendChild(avatarDiv);
        messageDiv.appendChild(wrapperDiv);
        
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
        
        // Store result
        this.currentMessages.push({ 
            content, 
            type: 'assistant', 
            timestamp: new Date(),
            isResult: true 
        });
    }
    
    downloadResult(content) {
        const blob = new Blob([content], { type: 'text/markdown' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `research-proposal-${Date.now()}.md`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
    
    // Utility functions
    markdownToHtml(markdown) {
        let processedMarkdown = markdown.replace(/^## (.*$)/gm, '<!-- section-break -->\n<h2>$1</h2>');
        const sections = processedMarkdown.split('<!-- section-break -->');
        
        const processedSections = sections.map(section => {
            let processed = section
                .replace(/^# (.*$)/gm, '<h1>$1</h1>')
                .replace(/^### (.*$)/gm, '<h3>$1</h3>')
                .replace(/^#### (.*$)/gm, '<h4>$1</h4>')
                .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/\*(.*?)\*/g, '<em>$1</em>')
                .replace(/^\d+\.\s+(.*)$/gm, '<li>$1</li>')
                .replace(/^\* (.*$)/gm, '<li>$1</li>');
            
            // Wrap consecutive <li> elements in <ol>
            processed = processed.replace(/(<li>.*<\/li>)/s, '<ol>$1</ol>');
            processed = processed.replace(/<\/ol>\s*<ol>/g, '');
            
            return processed;
        });
        
        return processedSections
            .join('')
            .replace(/\n\n/g, '</p><p>')
            .replace(/^(?!<[h|u|l|o])(.+)$/gm, '<p>$1</p>')
            .replace(/<p><\/p>/g, '');
    }
    
    // Error handling
    showError(message) {
        this.errorMessage.textContent = message;
        this.errorModal.style.display = 'flex';
    }
    
    hideError() {
        this.errorModal.style.display = 'none';
    }
    
    retryLastAction() {
        this.hideError();
    }
}

// Fill example prompt function
function fillPrompt(text) {
    const messageInput = document.getElementById('messageInput');
    const sendBtn = document.getElementById('sendBtn');
    
    // Check if access key is verified
    if (!window.researchUI || !window.researchUI.isAccessVerified) {
        // Open settings to prompt for access key
        if (window.researchUI) {
            window.researchUI.toggleSettings();
        }
        return;
    }
    
    messageInput.value = text;
    messageInput.focus();
    sendBtn.disabled = false;
    
    // Trigger input event to resize textarea
    messageInput.dispatchEvent(new Event('input'));
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    window.researchUI = new ResearchAgentUI();
});