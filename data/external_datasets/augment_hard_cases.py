#!/usr/bin/env python3
"""
===============================================================================
QUANTARA - Hard Case Augmentation for Edge Case Accuracy
===============================================================================
Targeted augmentation for emotions that the benchmark consistently misclassifies.
Focuses on:
  - Sarcasm/irony detection (frustration, contempt)
  - Subtle low-arousal emotions (boredom, calm, neutral, nostalgia)
  - Confused pairs: sadness vs worry, anger vs excitement, calm vs joy
  - Underperforming: grief, shame, overwhelmed, mindfulness, surprise, pride

Each emotion gets diverse, unambiguous examples designed to create clear
decision boundaries in the embedding space.

Usage:
    python data/external_datasets/augment_hard_cases.py
    python train_emotion_classifier.py --use-sentence-transformer \
        --external-data data/external_datasets/augmented_emotion_data.csv

===============================================================================
"""

import random
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_PATH = PROJECT_ROOT / 'data' / 'external_datasets' / 'augmented_emotion_data.csv'
OUTPUT_PATH = INPUT_PATH  # overwrite with appended hard cases

BIOMETRIC_RANGES = {
    'joy':          {'hr': (70, 90),   'hrv': (50, 80),  'eda': (2, 4)},
    'excitement':   {'hr': (85, 110),  'hrv': (40, 60),  'eda': (4, 7)},
    'enthusiasm':   {'hr': (75, 95),   'hrv': (45, 70),  'eda': (3, 5)},
    'fun':          {'hr': (75, 95),   'hrv': (50, 75),  'eda': (2, 4)},
    'gratitude':    {'hr': (65, 80),   'hrv': (60, 85),  'eda': (1, 3)},
    'pride':        {'hr': (70, 90),   'hrv': (45, 65),  'eda': (3, 5)},
    'sadness':      {'hr': (55, 70),   'hrv': (40, 60),  'eda': (1, 2)},
    'grief':        {'hr': (50, 70),   'hrv': (30, 50),  'eda': (1, 3)},
    'boredom':      {'hr': (55, 65),   'hrv': (55, 75),  'eda': (0.5, 1.5)},
    'nostalgia':    {'hr': (60, 75),   'hrv': (45, 65),  'eda': (1.5, 3)},
    'anger':        {'hr': (85, 110),  'hrv': (20, 40),  'eda': (5, 8)},
    'frustration':  {'hr': (80, 100),  'hrv': (25, 45),  'eda': (4, 7)},
    'hate':         {'hr': (85, 105),  'hrv': (20, 35),  'eda': (5, 9)},
    'contempt':     {'hr': (75, 90),   'hrv': (30, 50),  'eda': (3, 5)},
    'disgust':      {'hr': (70, 90),   'hrv': (35, 55),  'eda': (4, 7)},
    'jealousy':     {'hr': (80, 100),  'hrv': (25, 45),  'eda': (5, 8)},
    'fear':         {'hr': (80, 105),  'hrv': (25, 45),  'eda': (6, 10)},
    'anxiety':      {'hr': (75, 100),  'hrv': (20, 40),  'eda': (5, 9)},
    'worry':        {'hr': (70, 90),   'hrv': (30, 50),  'eda': (3, 6)},
    'overwhelmed':  {'hr': (85, 110),  'hrv': (15, 35),  'eda': (7, 12)},
    'stressed':     {'hr': (80, 105),  'hrv': (20, 40),  'eda': (5, 9)},
    'love':         {'hr': (65, 85),   'hrv': (55, 75),  'eda': (2, 4)},
    'compassion':   {'hr': (60, 80),   'hrv': (60, 80),  'eda': (1.5, 3)},
    'calm':         {'hr': (55, 70),   'hrv': (65, 90),  'eda': (0.5, 2)},
    'relief':       {'hr': (60, 80),   'hrv': (55, 80),  'eda': (2, 4)},
    'mindfulness':  {'hr': (55, 68),   'hrv': (70, 95),  'eda': (0.5, 1.5)},
    'resilience':   {'hr': (60, 75),   'hrv': (60, 85),  'eda': (1, 3)},
    'hope':         {'hr': (65, 80),   'hrv': (55, 75),  'eda': (1.5, 3)},
    'guilt':        {'hr': (70, 90),   'hrv': (30, 50),  'eda': (4, 7)},
    'shame':        {'hr': (75, 95),   'hrv': (25, 45),  'eda': (5, 8)},
    'surprise':     {'hr': (75, 100),  'hrv': (35, 55),  'eda': (4, 7)},
    'neutral':      {'hr': (60, 80),   'hrv': (50, 70),  'eda': (1, 3)},
}

# ─── Hard case templates: focused on benchmark failure modes ─────────────────

HARD_CASES = {
    # === SARCASM: benchmark gets these completely wrong ===
    'frustration': [
        "Oh great, another meeting that could have been an email",
        "Sure, let's have yet another pointless discussion about this",
        "Wonderful, the printer is jammed again for the fifth time today",
        "How lovely, they changed the requirements at the last minute again",
        "Perfect timing, the system crashed right before the deadline",
        "Nothing I love more than doing the same task over and over",
        "Ah yes, because that worked so well the last three times we tried it",
        "I'm so thrilled to redo all this work because someone changed their mind",
        "What a surprise, the meeting ran over by an hour again",
        "Just what I needed, more bureaucratic paperwork",
        "I'm so fed up with this broken process that never gets fixed",
        "This is incredibly frustrating, nothing works the way it should",
        "I've tried everything and nothing is working, I'm at my wit's end",
        "The constant delays are driving me absolutely crazy",
        "I keep hitting the same wall over and over and I'm losing patience",
        "Why does this always happen right when things start going well",
        "I've been on hold for two hours and I'm about to lose it",
        "The traffic is unbearable and I'm going to be late again",
        "They keep ignoring my suggestions and it's infuriating",
        "I'm frustrated beyond words with this entire situation",
        "Every time I fix one thing, two more problems appear",
        "This software crashes every single time I try to save",
        "I asked for help three times and nobody responded",
        "The WiFi drops out every five minutes and I can't work like this",
        "I'm so frustrated I could scream into a pillow",
    ],
    'contempt': [
        "Sure, that's just wonderful, like I needed more problems",
        "Oh, how impressive, they managed to do the bare minimum",
        "What a brilliant idea, I'm sure that will solve everything",
        "They think they're so clever but they have no idea what they're doing",
        "How charming, another person who thinks they know better than everyone",
        "I have zero respect for people who take credit for others' work",
        "Their incompetence is truly remarkable at this point",
        "I look down on anyone who treats service workers poorly",
        "The way they dismissed everyone's concerns shows their true character",
        "They don't deserve the position they hold, it's laughable",
        "I find their arrogance absolutely pathetic",
        "The audacity of thinking they're above the rules disgusts me",
        "What a joke, they actually think that excuse is believable",
        "I have nothing but disdain for that kind of behavior",
        "Their condescending attitude makes my skin crawl",
        "I scoff at their feeble attempt to appear competent",
        "The sheer mediocrity of their effort is almost impressive",
        "I can't take seriously anyone who behaves like that",
        "They're beneath my consideration at this point",
        "How pathetic that they think manipulation goes unnoticed",
    ],

    # === CLEAR NEGATIVE EMOTIONS: sadness gets confused with worry ===
    'sadness': [
        "I feel so sad and empty inside, like nothing matters anymore",
        "Tears keep falling and I can't make them stop",
        "This is the worst day of my life, everything feels hopeless",
        "A deep sadness has settled over me and I can't shake it",
        "I feel like crying but I don't even have the energy for tears",
        "My heart is heavy with sorrow today",
        "I'm deeply saddened by what happened and can't stop thinking about it",
        "There's an ache in my chest that won't go away, pure sadness",
        "I feel dejected and downcast, the world looks grey",
        "The sadness is overwhelming, I just want to curl up and disappear",
        "I'm heartbroken and don't know how to pick up the pieces",
        "Everything beautiful reminds me of what I've lost and makes me sad",
        "I'm drowning in sorrow and can't see a way out",
        "The melancholy is thick today, I feel blue and listless",
        "Sadness washes over me in waves that I can't control",
        "I feel so down that even sunshine can't lift my spirits",
        "My soul aches with a profound sadness I can't explain",
        "I'm sad to the bone, nothing brings me comfort right now",
        "The world feels darker and colder when you're this sad",
        "I just feel really really sad and I don't know why",
    ],

    # === ANGER: gets confused with excitement ===
    'anger': [
        "I am absolutely furious right now, this is unacceptable",
        "I'm so angry I'm shaking, how dare they do this",
        "My blood is boiling with rage at this injustice",
        "I want to punch a wall because I'm so mad",
        "This makes me livid, I've never been this angry in my life",
        "I'm seething with anger at their betrayal",
        "The rage I feel is consuming every rational thought",
        "I'm enraged by their lies and deception",
        "I am incandescent with fury right now",
        "How dare they treat people like that, I'm furious",
        "I'm so angry I can barely form coherent sentences",
        "The disrespect has pushed me past my breaking point",
        "I want to scream at the top of my lungs from this anger",
        "My jaw is clenched so hard from the rage I feel",
        "I'm absolutely irate about the unfair treatment",
        "This outrageous behavior has me seeing red",
        "I'm channeling all my anger into this confrontation",
        "The injustice fills me with white-hot anger",
        "I've been wronged and I'm not going to take it quietly",
        "I'm furious, hostile, and ready to make my feelings known",
    ],

    # === CALM: gets confused with joy ===
    'calm': [
        "I feel completely at peace, my mind is quiet and still",
        "Everything is serene and tranquil right now",
        "I'm in a state of deep calm, no worries, no rush",
        "The stillness around me matches the peace inside me",
        "I feel settled and composed, nothing can disturb me",
        "My breathing is slow and even, I'm perfectly relaxed",
        "I'm at ease with everything, no tension whatsoever",
        "A gentle calm has washed over me like a warm bath",
        "I feel placid and untroubled by anything today",
        "The serenity of this moment is beautiful in its simplicity",
        "I'm not happy or sad, just peacefully calm",
        "My mind is clear and undisturbed, like a still pond",
        "I feel a quiet contentment, not excitement, just peace",
        "Everything has slowed down and I feel centered",
        "I'm resting in a state of complete inner stillness",
        "There's no urgency, no stress, just peaceful calm",
        "I feel emotionally neutral and relaxed, at rest",
        "The calm I feel isn't joy, it's deeper than that — it's peace",
        "I'm sitting quietly, feeling nothing but gentle tranquility",
        "My nervous system has settled into a deeply calm state",
    ],

    # === NEUTRAL: gets confused with worry ===
    'neutral': [
        "I guess it's fine, whatever happens happens",
        "I don't feel strongly about it one way or another",
        "It's just a regular day, nothing special about it",
        "I'm neither happy nor sad, just existing",
        "I have no particular opinion on the matter",
        "Things are okay, not great not terrible, just okay",
        "I feel indifferent about the whole situation",
        "It doesn't affect me emotionally one way or another",
        "I'm in a completely neutral state of mind right now",
        "No strong feelings here, everything is just baseline",
        "I acknowledge the situation without any emotional reaction",
        "My emotional state is flat, not depressed, just neutral",
        "I can take it or leave it, doesn't matter to me",
        "I'm emotionally detached from this topic entirely",
        "It is what it is, I feel nothing particular about it",
        "I'm observing without judgment or emotional investment",
        "My reaction is completely neutral, no charge at all",
        "I feel blank, not in a bad way, just emotionally level",
        "There's no emotional response, just factual acknowledgment",
        "I'm unmoved by this, it simply doesn't register emotionally",
    ],

    # === GRIEF: gets confused with surprise ===
    'grief': [
        "The pain of losing them is unbearable, I'm shattered",
        "I am consumed by grief, nothing else exists right now",
        "My world collapsed when they died, I'm devastated",
        "The grief hits me in waves so strong I can barely stand",
        "I'm mourning deeply, the loss is a wound that won't heal",
        "I've never known pain like this grief, it's crushing",
        "Every day without them is a reminder of what I've lost",
        "I'm grieving so deeply that I've forgotten how to function",
        "The absence is a constant, aching presence in my life",
        "I'm broken by this loss and I don't know how to be whole again",
        "The depth of my grief surprises me with how physical it is",
        "I carry this grief like a stone in my chest every single day",
        "Nothing prepares you for the devastating weight of true grief",
        "I'm in deep mourning and the world feels completely empty",
        "This grief is the heaviest emotion I have ever carried",
        "I am inconsolable, the tears and pain never fully stop",
        "Bereavement has changed me as a person, I grieve constantly",
        "The loss left a void that nothing and no one can fill",
        "I'm wrecked with grief, barely able to get through the day",
        "This devastating loss has plunged me into profound grief",
    ],

    # === SHAME: gets confused with relief ===
    'shame': [
        "I want to disappear, everyone knows what I did wrong",
        "The shame I feel is suffocating, I can't face anyone",
        "I'm deeply ashamed of myself and my terrible behavior",
        "I feel so exposed and humiliated, like everyone is staring",
        "The shame burns inside me every time I think about it",
        "I'm mortified and want to crawl into a hole and never come out",
        "I feel worthless and undeserving after what I did",
        "The public embarrassment has left me paralyzed with shame",
        "I'm consumed by self-loathing and deep shame",
        "Everyone's judgment makes the shame even more unbearable",
        "I feel like I've disgraced myself and my family",
        "The humiliation replays in my mind on an endless loop",
        "I'm so ashamed I can't even look at my own reflection",
        "Being caught in that lie was the most shameful experience",
        "I feel fundamentally broken and unworthy as a person",
        "The shame spiral has me questioning everything about myself",
        "I'd rather be invisible than face the shame I feel right now",
        "I'm drowning in self-disgust and deep personal shame",
        "The weight of my shame makes every interaction painful",
        "I carry this shame like a brand that everyone can see",
    ],

    # === OVERWHELMED: gets confused with guilt ===
    'overwhelmed': [
        "There's too much happening at once and I'm falling apart",
        "I am completely overwhelmed, my brain is shutting down",
        "Everything is crashing down on me and I can't cope",
        "I'm drowning under the weight of all these responsibilities",
        "The sheer volume of everything is too much for me to handle",
        "I feel like I'm being buried alive under all this pressure",
        "My plate is so full it's breaking and I can't manage any of it",
        "I'm at absolute capacity, one more thing will break me",
        "The demands from every direction are overwhelming me completely",
        "I can't process anything anymore because there's just too much",
        "I'm overwhelmed to the point of complete paralysis",
        "Everything needs attention right now and I can't do it all",
        "The amount of work ahead of me is genuinely overwhelming",
        "I feel crushed under the mountain of things to deal with",
        "Sensory overload combined with emotional overload is wrecking me",
        "I'm so overwhelmed I literally cannot decide what to do first",
        "My brain is full, my heart is full, everything is too much",
        "I need everyone to stop asking things of me, I'm overwhelmed",
        "The complexity of this situation is beyond what I can handle",
        "I'm past my limit, I am thoroughly and completely overwhelmed",
    ],

    # === SURPRISE: currently overpredicted, needs REAL surprise examples ===
    'surprise': [
        "Wait, what?! I did not see that coming at all!",
        "I'm absolutely shocked, I never expected this to happen",
        "My jaw literally dropped when I heard the news",
        "I can't believe what just happened, I'm stunned",
        "That was completely unexpected, I'm in total disbelief",
        "No way! Are you serious? I'm so surprised right now",
        "I was caught completely off guard by that announcement",
        "Well that was a twist I never could have predicted",
        "I'm genuinely taken aback, this changes everything",
        "The plot twist blew my mind, I'm still processing it",
        "I'm flabbergasted, that's the last thing I expected",
        "Out of nowhere, everything changed and I'm still in shock",
        "I had no idea that was coming, I'm completely blindsided",
        "That revelation was startling, I need a moment to process",
        "I'm wide-eyed with surprise, I genuinely did not know",
        "What a shocking turn of events, nobody saw this coming",
        "I'm dumbfounded by this unexpected development",
        "The surprise announcement left everyone speechless",
        "I'm astonished, this is the exact opposite of what I expected",
        "That caught me totally off guard, I'm still recovering from the shock",
    ],

    # === MINDFULNESS: gets confused with fear ===
    'mindfulness': [
        "I'm focusing on my breath, inhale deeply, exhale slowly",
        "Right now I'm fully present, noticing every sensation in my body",
        "I'm practicing mindful awareness of this current moment",
        "I observe my thoughts drifting by like clouds, without attachment",
        "I'm grounded in the here and now, fully aware and accepting",
        "Each breath anchors me deeper into present-moment awareness",
        "I notice the weight of my body in the chair, the air on my skin",
        "I'm doing a body scan, bringing gentle attention to each area",
        "Mindful eating: tasting each bite, noticing the texture and flavor",
        "I'm aware of the sounds around me without labeling them good or bad",
        "I practice non-judgmental observation of my inner experience",
        "My attention rests gently on the present, not past or future",
        "I'm cultivating awareness: noting what arises without reacting",
        "I feel the ground beneath my feet, solid and supporting me",
        "I breathe mindfully, aware of the rise and fall of my chest",
        "This walking meditation keeps me in the present step by step",
        "I acknowledge this emotion with mindful acceptance",
        "I'm sitting in stillness, observing my mind with curiosity",
        "My practice today is simply being present with what is",
        "I'm mindfully noticing my reactions without being controlled by them",
    ],

    # === PRIDE: gets confused with worry ===
    'pride': [
        "I did it! I actually achieved what everyone said was impossible",
        "I'm beaming with pride at what we accomplished together",
        "This is my greatest achievement and I'm so proud of myself",
        "I feel immensely proud of the hard work that led to this moment",
        "I earned this through years of dedication and I'm proud",
        "Watching my child graduate fills me with overwhelming pride",
        "I look at what I've built and I swell with pride",
        "I'm proud of who I've become through all these challenges",
        "My team delivered an incredible result and I couldn't be prouder",
        "Standing on that podium was the proudest moment of my life",
        "I take great pride in the quality of my craftsmanship",
        "Against all odds, I succeeded, and I'm filled with pride",
        "I'm proud to represent my community on this stage",
        "This award validates everything I've worked toward, I'm so proud",
        "I held my head high because I know I gave it everything",
        "The sense of accomplishment fills me with deep pride",
        "I'm proud not just of the result, but of the journey",
        "I finally proved everyone wrong and it feels incredible",
        "I can't help but feel proud when I see how far I've come",
        "This breakthrough is something I'll always be proud of",
    ],

    # === BOREDOM: gets confused with shame ===
    'boredom': [
        "I'm so bored I could watch paint dry for entertainment",
        "There's nothing to do and time is moving at a crawl",
        "This is the most tedious and boring task I've ever done",
        "I'm dying of boredom here, someone please save me",
        "Every minute feels like an hour when you're this bored",
        "I've run out of things to entertain myself with",
        "The monotony of this routine is killing me with boredom",
        "I'm so understimulated that my brain has turned to mush",
        "Watching this presentation is like watching grass grow",
        "I have zero interest in anything right now, total boredom",
        "Same old same old, the repetition is mind-numbing",
        "I keep looking at the clock because I'm bored out of my skull",
        "There is not a single interesting thing happening right now",
        "The tedium of this waiting room is excruciating",
        "I've scrolled through everything and there's nothing left",
        "Boredom has set in hard, I can't muster interest in anything",
        "This day is dragging on with absolutely nothing engaging",
        "I'm restless from pure boredom, not anxiety, just boredom",
        "The lecture is so dull I've started counting floor tiles",
        "I would literally do anything to escape this boredom right now",
    ],

    # === NOSTALGIA: gets confused with joy ===
    'nostalgia': [
        "I miss those days so much, they'll never come back",
        "Looking at old photos makes my heart ache for the past",
        "I long for the simpler times when we were all together",
        "That song takes me back to a time I can never return to",
        "The bittersweet feeling of remembering what's gone forever",
        "I'm nostalgic for my childhood, those innocent years",
        "Walking past our old hangout spot fills me with wistful longing",
        "I miss who I used to be, those memories feel like another life",
        "The smell of autumn leaves takes me back to a time long gone",
        "I'm aching for the past, knowing I can never go back there",
        "These memories are precious but painful because they're over",
        "I feel a deep yearning for the way things used to be",
        "Everything reminds me of a time that exists only in memory now",
        "The nostalgia hits different when you realize you were happy then",
        "I cling to these old memories because they're all I have left",
        "Time has passed and I miss what was, that bittersweet nostalgia",
        "I'm revisiting old places and feeling the ache of nostalgia",
        "Those golden days are gone and all that's left is this longing",
        "Nostalgia washes over me like a gentle but sad wave",
        "I wish I could freeze time and go back to those moments",
    ],

    # === GRATITUDE: gets confused with relief ===
    'gratitude': [
        "Thank you from the bottom of my heart, you changed my life",
        "I am so incredibly grateful for your kindness and support",
        "I don't know what I would have done without you, thank you",
        "My heart overflows with gratitude for all you've done",
        "I'm deeply thankful for this opportunity and I won't waste it",
        "Your generosity fills me with immense gratitude",
        "I appreciate everything you've sacrificed to help me",
        "I feel blessed and grateful for the people in my life",
        "Thank you for believing in me when nobody else did",
        "I'm grateful for every single day and every small kindness",
        "The depth of my gratitude is hard to put into words",
        "I owe you so much, I'm truly and deeply grateful",
        "I wake up grateful for the life I get to live",
        "Your support means the world to me, I'm so thankful",
        "I'm filled with appreciation for this wonderful community",
        "Gratitude is what I feel most strongly right now",
        "I never take for granted the blessings I've received",
        "Thank you for being there during my darkest hours",
        "I'm grateful not just for what you did but for who you are",
        "My heart is full of genuine thankfulness and appreciation",
    ],
}

# How many copies of each hard case template to include
COPIES_PER_TEMPLATE = 3


def generate_biometrics(emotion: str) -> dict:
    ranges = BIOMETRIC_RANGES.get(emotion, BIOMETRIC_RANGES['neutral'])
    return {
        'hr': round(random.uniform(*ranges['hr']), 1),
        'hrv': round(random.uniform(*ranges['hrv']), 1),
        'eda': round(random.uniform(*ranges['eda']), 2),
    }


def main():
    print("=" * 65)
    print("  QUANTARA - Hard Case Augmentation")
    print("=" * 65)

    random.seed(42)
    np.random.seed(42)

    # Load existing augmented data
    if INPUT_PATH.exists():
        df_existing = pd.read_csv(INPUT_PATH)
        print(f"\n  Existing data: {len(df_existing)} rows")
    else:
        print(f"\n  [ERROR] Input not found: {INPUT_PATH}")
        print("  Run augment_balanced.py first.")
        return

    # Generate hard case rows
    hard_rows = []
    for emotion, templates in HARD_CASES.items():
        for template in templates:
            for _ in range(COPIES_PER_TEMPLATE):
                bio = generate_biometrics(emotion)
                hard_rows.append({
                    'text': template,
                    'emotion': emotion,
                    'hr': bio['hr'],
                    'hrv': bio['hrv'],
                    'eda': bio['eda'],
                })

    df_hard = pd.DataFrame(hard_rows)

    print(f"  Hard case samples: {len(df_hard)}")
    print(f"\n  Hard case distribution:")
    print("  " + "-" * 40)
    for emotion, count in sorted(Counter(df_hard['emotion']).items(), key=lambda x: -x[1]):
        print(f"    {emotion:<15} {count:4d}")

    # Combine
    df_combined = pd.concat([df_existing, df_hard], ignore_index=True)
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save
    df_combined.to_csv(OUTPUT_PATH, index=False)

    print(f"\n  Combined total: {len(df_combined)}")
    print(f"  Saved to: {OUTPUT_PATH}")

    print("\n" + "=" * 65)
    print("  Done! Retrain with:")
    print(f"    python train_emotion_classifier.py --use-sentence-transformer \\")
    print(f"        --external-data {OUTPUT_PATH}")
    print("=" * 65)


if __name__ == '__main__':
    main()
