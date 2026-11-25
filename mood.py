import math

class MoodCircle:
    def __init__(self, x=0.0, y=0.0, inertia=0.45, decay=0.05):
        """
        x: Valence (-1.0 negative to 1.0 positive)
        y: Arousal (-1.0 low energy to 1.0 high energy)
        inertia: How strongly new events push the mood (0.1 = stubborn, 0.9 = volatile)
        decay: How fast the bot returns to neutral (0,0) each turn
        """
        self.x = x
        self.y = y
        self.inertia = inertia
        self.decay = decay

    def update(self, dx: float, dy: float):
        """Nudges the mood vector by (dx, dy) and keeps it inside the unit circle."""
        # 1. Apply natural decay back toward neutral (0, 0)
        self.x *= (1.0 - self.decay)
        self.y *= (1.0 - self.decay)

        # 2. Apply the new emotional impulse scaled by inertia
        self.x += dx * self.inertia
        self.y += dy * self.inertia

        # 3. Clamp to the unit circle (radius <= 1.0)
        magnitude = math.hypot(self.x, self.y)
        if magnitude > 1.0:
            self.x /= magnitude
            self.y /= magnitude

    def get_mood_state(self):
        """Converts (x, y) coordinates into an angle, intensity, and label."""
        intensity = math.hypot(self.x, self.y)
        
        # Deadzone in the center of the circle = Neutral
        if intensity < 0.2:
            return "Neutral", intensity

        # atan2 returns angle in radians from -pi to pi; convert to 0-360 degrees
        angle = math.degrees(math.atan2(self.y, self.x)) % 360

        # Map 8 slices (45 degrees each) around the circle
        if 337.5 <= angle or angle < 22.5:
            mood = "Happy / Pleased"
        elif 22.5 <= angle < 67.5:
            mood = "Excited / Enthusiastic"
        elif 67.5 <= angle < 112.5:
            mood = "Alert / Hyper"
        elif 112.5 <= angle < 157.5:
            mood = "Angry / Frustrated"
        elif 157.5 <= angle < 202.5:
            mood = "Upset / Displeased"
        elif 202.5 <= angle < 247.5:
            mood = "Sad / Depressed"
        elif 247.5 <= angle < 292.5:
            mood = "Tired / Bored"
        else: # 292.5 <= angle < 337.5
            mood = "Calm / Relaxed"

        print(mood, round(intensity,2))
        return mood, round(intensity, 2)

    def get_system_prompt(self):
        MOOD_TO_FISH_TAGS = {
            "Happy / Pleased": ["[happy]", "[satisfied]", "[delighted]"],
            "Excited / Enthusiastic": ["[excited]", "[hopeful]", "[optimistic]"],
            "Alert / Hyper": ["[excited]", "[curious]"],
            "Angry / Frustrated": ["[angry]", "[frustrated]", "[upset]"],
            "Upset / Displeased": ["[upset]", "[disappointed]", "[unhappy]"],
            "Sad / Depressed": ["[sad]", "[depressed]", "[resigned]"],
            "Tired / Bored": ["[bored]", "[calm]", "[indifferent]"],
            "Calm / Relaxed": ["[calm]", "[relaxed]"],
            "Neutral": ["[calm]", "[relaxed]"],
        }
        """Generates the dynamic instruction to feed into your LLM."""
        mood, intensity = self.get_mood_state()

        # Scale how strongly the tone instruction is phrased based on intensity (0.0 to 1.0)
        if intensity < 0.35:
            tone_instruction = "Subtly reflect this tone and energy level in your response."
        elif intensity < 0.7:
            tone_instruction = (
                "Clearly reflect this tone and energy level in your word choice, "
                "pacing, and punctuation."
            )
        else:
            tone_instruction = (
                "Strongly express this tone and energy level through distinct word choice, "
                "sentence rhythm, and punctuation while staying in character."
            )

        # Suggest Fish Audio [bracket] emotion tags once outside the Neutral deadzone (< 0.2)
        suggested_tags = ", ".join(MOOD_TO_FISH_TAGS.get(mood, ["[calm]"]))
        if intensity < 0.2:
            tag_instruction = ""
        elif intensity < 0.5:
            tag_instruction = (
                f" You may optionally place one tag like {suggested_tags} before a phrase "
                f"when it fits naturally."
            )
        else:
            tag_instruction = (
                f" Use a matching expression tag such as {suggested_tags} before a phrase "
                f"to shape spoken delivery, without putting tags on every single clause."
            )

        return (
            f"Current emotional state: {mood} (Intensity: {intensity}/1.0, "
            f"Valence: {self.x:.2f}, Arousal: {self.y:.2f}). "
            f"{tone_instruction}{tag_instruction}"
        )
# --- Example Usage ---
bot_mood = MoodCircle(inertia=0.4)

# User sends an annoying or aggressive message -> Negative Valence (-0.8), High Arousal (+0.7)
bot_mood.update(dx=-0.8, dy=0.7)
print(bot_mood.get_system_prompt())

# User apologizes and sends something nice -> Positive Valence (+0.9), Calm Arousal (-0.3)
bot_mood.update(dx=0.9, dy=-0.3)
print(bot_mood.get_system_prompt())