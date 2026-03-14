/**
 * ===============================================================================
 * QUANTARA NEURAL ECOSYSTEM - Emotion GPT Connector (32-Emotion Taxonomy)
 * ===============================================================================
 * JavaScript connector for integrating Emotion GPT API with Neural Ecosystem.
 * Supports 32 emotions across 9 families with hierarchical classification.
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

// ─── 32-Emotion Taxonomy ────────────────────────────────────────────────────

const EMOTION_FAMILIES = {
    Joy: ['joy', 'excitement', 'enthusiasm', 'fun', 'gratitude', 'pride'],
    Sadness: ['sadness', 'grief', 'boredom', 'nostalgia'],
    Anger: ['anger', 'frustration', 'hate', 'contempt', 'disgust', 'jealousy'],
    Fear: ['fear', 'anxiety', 'worry', 'overwhelmed', 'stressed'],
    Love: ['love', 'compassion'],
    Calm: ['calm', 'relief', 'mindfulness', 'resilience', 'hope'],
    'Self-Conscious': ['guilt', 'shame'],
    Surprise: ['surprise'],
    Neutral: ['neutral'],
};

const ALL_EMOTIONS = Object.values(EMOTION_FAMILIES).flat();

// Reverse lookup: emotion → family
const EMOTION_TO_FAMILY = {};
for (const [family, emotions] of Object.entries(EMOTION_FAMILIES)) {
    for (const emotion of emotions) {
        EMOTION_TO_FAMILY[emotion] = family;
    }
}


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
     * Get list of supported emotions (32)
     */
    async getEmotions() {
        const response = await fetch(`${this.baseUrl}/api/emotion/emotions`);
        return response.json();
    }

    /**
     * Get all emotion families with their emotions
     */
    async getEmotionFamilies() {
        const response = await fetch(`${this.baseUrl}/api/emotion/family`);
        return response.json();
    }

    /**
     * Get emotions in a specific family
     * @param {string} familyName - Family name (Joy, Sadness, Anger, etc.)
     */
    async getEmotionFamily(familyName) {
        const response = await fetch(`${this.baseUrl}/api/emotion/family/${encodeURIComponent(familyName)}`);
        return response.json();
    }

    /**
     * Analyze emotional content of text
     * @param {string} text - Text to analyze
     * @param {Object} biometrics - Optional biometric data
     * @returns {Object} Analysis result with emotion, family, confidence, is_fallback
     */
    async analyze(text, biometrics = null) {
        const body = { text };
        if (biometrics) body.biometrics = biometrics;

        const response = await fetch(`${this.baseUrl}/api/emotion/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        return response.json();
    }

    /**
     * Generate emotion-aware text
     * @param {string} prompt - Starting prompt
     * @param {string} emotion - Target emotion (any of 32, optional)
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
     * @param {string} emotion - Current emotion (any of 32)
     * @returns {Object} technique, transition pathway, coaching prompt
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
     * Get emotion transition pathway
     * @param {string} fromEmotion - Current emotion
     * @param {string} toEmotion - Target emotion (optional)
     * @returns {Object} transition method, technique, coaching prompt
     */
    async getEmotionTransition(fromEmotion, toEmotion = null) {
        const response = await fetch(`${this.baseUrl}/api/neural/emotion-transition`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ from_emotion: fromEmotion, to_emotion: toEmotion })
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
 * Connects Emotion GPT (32 emotions) to all Quantara services
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
     * Process user message through emotion pipeline (32 emotions)
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

        // Get therapy technique for negative-valence families
        const negativeValenceFamilies = ['Sadness', 'Anger', 'Fear', 'Self-Conscious'];
        let therapy = null;
        let transition = null;

        if (negativeValenceFamilies.includes(analysis.family)) {
            therapy = await this.emotionAPI.getTherapyTechnique(analysis.dominant_emotion);
            transition = await this.emotionAPI.getEmotionTransition(analysis.dominant_emotion);
        }

        // Emit event for workflow engine
        this.emit('emotion_detected', {
            emotion: analysis.dominant_emotion,
            family: analysis.family,
            confidence: analysis.confidence,
            is_fallback: analysis.is_fallback,
            message,
            timestamp: new Date().toISOString()
        });

        return {
            analysis,
            coaching,
            therapy,
            transition,
            workflowTriggered: true
        };
    }

    /**
     * Process biometric data for emotion correlation (family-aware)
     * Integrates with Biometric Integration Engine
     */
    async processBiometricData(biometricData, recentText = null) {
        const { heart_rate, hrv, eda } = biometricData;

        // Family-level biometric inference
        let inferredEmotion = 'neutral';
        let inferredFamily = 'Neutral';

        if (heart_rate > 100 && hrv < 30 && eda > 5) {
            // High arousal + low HRV + high EDA
            if (eda > 7) {
                inferredEmotion = 'overwhelmed';
                inferredFamily = 'Fear';
            } else {
                inferredEmotion = 'fear';
                inferredFamily = 'Fear';
            }
        } else if (heart_rate > 90 && hrv < 40 && eda > 5) {
            inferredEmotion = 'anger';
            inferredFamily = 'Anger';
        } else if (heart_rate > 90 && hrv > 50) {
            inferredEmotion = 'excitement';
            inferredFamily = 'Joy';
        } else if (heart_rate < 65 && hrv < 40) {
            inferredEmotion = 'sadness';
            inferredFamily = 'Sadness';
        } else if (heart_rate < 70 && hrv > 65) {
            inferredEmotion = 'calm';
            inferredFamily = 'Calm';
        } else if (heart_rate > 70 && hrv < 45 && eda > 4) {
            inferredEmotion = 'guilt';
            inferredFamily = 'Self-Conscious';
        }

        // Cross-reference with text if available
        let textAnalysis = null;
        if (recentText) {
            textAnalysis = await this.emotionAPI.analyze(recentText);

            // Weight text emotion more heavily if biometric is ambiguous
            if (textAnalysis.confidence > 0.5) {
                inferredEmotion = textAnalysis.dominant_emotion;
                inferredFamily = textAnalysis.family || EMOTION_TO_FAMILY[inferredEmotion] || 'Neutral';
            }
        }

        return {
            biometricInference: inferredEmotion,
            biometricFamily: inferredFamily,
            textAnalysis,
            biometricData,
            correlationConfidence: textAnalysis ?
                (textAnalysis.dominant_emotion === inferredEmotion ? 'high' : 'moderate') : 'biometric_only'
        };
    }

    /**
     * Generate real-time dashboard data (family-aware)
     * Integrates with Dashboard Data systems
     */
    async getDashboardData(userId, timeRange = '24h') {
        return {
            userId,
            timeRange,
            emotionFamilies: Object.keys(EMOTION_FAMILIES),
            emotionTrend: [],
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
    module.exports = {
        QuantaraEmotionAPI,
        NeuralEmotionIntegration,
        EMOTION_FAMILIES,
        ALL_EMOTIONS,
        EMOTION_TO_FAMILY
    };
}

// Export for ES modules
if (typeof window !== 'undefined') {
    window.QuantaraEmotionAPI = QuantaraEmotionAPI;
    window.NeuralEmotionIntegration = NeuralEmotionIntegration;
    window.EMOTION_FAMILIES = EMOTION_FAMILIES;
    window.ALL_EMOTIONS = ALL_EMOTIONS;
    window.EMOTION_TO_FAMILY = EMOTION_TO_FAMILY;
}
