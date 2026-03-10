/**
 * ===============================================================================
 * QUANTARA NEURAL ECOSYSTEM - Emotion GPT Connector
 * ===============================================================================
 * JavaScript connector for integrating Emotion GPT API with Neural Ecosystem.
 *
 * Integration Points:
 * - Neural Workflow AI Engine
 * - AI Conversational Coach
 * - Emotion-Aware Training Engine
 * - Psychology Emotion Database
 * - Biometric Integration Engine
 * - Therapist Dashboard Engine
 * - Real-time Dashboard Data
 *
 * Usage:
 *   const emotionAPI = new QuantaraEmotionAPI('http://localhost:5050');
 *   const analysis = await emotionAPI.analyze('I feel happy today');
 * ===============================================================================
 */

class QuantaraEmotionAPI {
    constructor(baseUrl = 'http://localhost:5050') {
        this.baseUrl = baseUrl;
        this.isConnected = false;
    }

    /**
     * Check API connection status
     */
    async checkStatus() {
        try {
            const response = await fetch(`${this.baseUrl}/api/emotion/status`);
            const data = await response.json();
            this.isConnected = data.status === 'online';
            return data;
        } catch (error) {
            this.isConnected = false;
            throw new Error(`Emotion API not available: ${error.message}`);
        }
    }

    /**
     * Get list of supported emotions
     */
    async getEmotions() {
        const response = await fetch(`${this.baseUrl}/api/emotion/emotions`);
        return response.json();
    }

    /**
     * Analyze emotional content of text
     * @param {string} text - Text to analyze
     * @returns {Object} Analysis result with scores and dominant emotion
     */
    async analyze(text) {
        const response = await fetch(`${this.baseUrl}/api/emotion/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });
        return response.json();
    }

    /**
     * Generate emotion-aware text
     * @param {string} prompt - Starting prompt
     * @param {string} emotion - Target emotion (optional)
     * @param {Object} options - Generation options
     */
    async generate(prompt, emotion = null, options = {}) {
        const response = await fetch(`${this.baseUrl}/api/emotion/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt,
                emotion,
                max_tokens: options.maxTokens || 150,
                temperature: options.temperature || 0.8
            })
        });
        return response.json();
    }

    /**
     * Get empathetic coaching response
     * @param {string} message - User message
     * @param {string} emotion - Detected emotion (optional, auto-detected if missing)
     * @param {Object} biometric - Biometric data (optional)
     */
    async coach(message, emotion = null, biometric = null) {
        const response = await fetch(`${this.baseUrl}/api/emotion/coach`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message, emotion, biometric })
        });
        return response.json();
    }

    /**
     * Get therapy technique recommendation
     * @param {string} emotion - Current emotion
     */
    async getTherapyTechnique(emotion) {
        const response = await fetch(`${this.baseUrl}/api/emotion/therapy`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ emotion })
        });
        return response.json();
    }

    /**
     * Trigger Neural Ecosystem workflow based on emotion
     * @param {string} text - Text to analyze
     * @param {Object} biometric - Biometric data
     */
    async triggerWorkflow(text, biometric = null) {
        const response = await fetch(`${this.baseUrl}/api/neural/emotion-workflow`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, biometric })
        });
        return response.json();
    }
}

/**
 * Neural Ecosystem Integration Module
 * Connects Emotion GPT to all Quantara services
 */
class NeuralEmotionIntegration {
    constructor(emotionApiUrl = 'http://localhost:5050') {
        this.emotionAPI = new QuantaraEmotionAPI(emotionApiUrl);
        this.listeners = new Map();
    }

    /**
     * Initialize connection to Emotion GPT
     */
    async initialize() {
        try {
            const status = await this.emotionAPI.checkStatus();
            console.log('[NeuralEmotion] Connected to Emotion GPT:', status);
            return true;
        } catch (error) {
            console.error('[NeuralEmotion] Failed to connect:', error);
            return false;
        }
    }

    /**
     * Process user message through emotion pipeline
     * Integrates with AI Conversational Coach
     */
    async processUserMessage(message, context = {}) {
        // Analyze emotion
        const analysis = await this.emotionAPI.analyze(message);

        // Get coaching response
        const coaching = await this.emotionAPI.coach(
            message,
            analysis.dominant_emotion,
            context.biometric
        );

        // Get therapy technique if needed
        let therapy = null;
        if (['sadness', 'anger', 'fear'].includes(analysis.dominant_emotion)) {
            therapy = await this.emotionAPI.getTherapyTechnique(analysis.dominant_emotion);
        }

        // Emit event for workflow engine
        this.emit('emotion_detected', {
            emotion: analysis.dominant_emotion,
            confidence: analysis.confidence,
            message,
            timestamp: new Date().toISOString()
        });

        return {
            analysis,
            coaching,
            therapy,
            workflowTriggered: true
        };
    }

    /**
     * Process biometric data for emotion correlation
     * Integrates with Biometric Integration Engine
     */
    async processBiometricData(biometricData, recentText = null) {
        const { heart_rate, hrv, eda, temperature } = biometricData;

        // Infer emotional state from biometrics
        let inferredEmotion = 'neutral';

        if (heart_rate > 100 && hrv < 30) {
            inferredEmotion = 'fear'; // High arousal, low variability
        } else if (heart_rate > 90 && hrv > 50) {
            inferredEmotion = 'joy'; // High arousal, good variability
        } else if (heart_rate < 65 && hrv < 40) {
            inferredEmotion = 'sadness'; // Low arousal, low variability
        }

        // Cross-reference with text if available
        let textAnalysis = null;
        if (recentText) {
            textAnalysis = await this.emotionAPI.analyze(recentText);

            // Weight text emotion more heavily if biometric is ambiguous
            if (textAnalysis.confidence > 0.5) {
                inferredEmotion = textAnalysis.dominant_emotion;
            }
        }

        return {
            biometricInference: inferredEmotion,
            textAnalysis,
            biometricData,
            correlationConfidence: textAnalysis ?
                (textAnalysis.dominant_emotion === inferredEmotion ? 'high' : 'moderate') : 'biometric_only'
        };
    }

    /**
     * Generate real-time dashboard data
     * Integrates with Dashboard Data systems
     */
    async getDashboardData(userId, timeRange = '24h') {
        // This would connect to your database
        // For now, return structure
        return {
            userId,
            timeRange,
            emotionTrend: [], // Would be populated from DB
            dominantEmotions: [],
            biometricCorrelations: [],
            coachingInteractions: [],
            lastUpdated: new Date().toISOString()
        };
    }

    /**
     * Event system for workflow integration
     */
    on(event, callback) {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, []);
        }
        this.listeners.get(event).push(callback);
    }

    emit(event, data) {
        if (this.listeners.has(event)) {
            this.listeners.get(event).forEach(cb => cb(data));
        }
    }
}

// Export for Node.js / CommonJS
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { QuantaraEmotionAPI, NeuralEmotionIntegration };
}

// Export for ES modules
if (typeof window !== 'undefined') {
    window.QuantaraEmotionAPI = QuantaraEmotionAPI;
    window.NeuralEmotionIntegration = NeuralEmotionIntegration;
}
