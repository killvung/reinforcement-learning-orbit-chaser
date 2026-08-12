export class ArcadeAudio {
  private context?: AudioContext;
  enabled = true;

  unlock(): void {
    this.context ??= new AudioContext();
    if (this.context.state === 'suspended') void this.context.resume();
  }

  pellet(): void { this.tone(700, 0.055, 'square', 0.045, 980); }
  surge(): void { this.tone(310, 0.11, 'triangle', 0.07, 620); window.setTimeout(() => this.tone(780, 0.14, 'triangle', 0.06, 1180), 75); }
  danger(): void { this.tone(150, 0.06, 'sine', 0.025, 135); }
  captured(): void { this.tone(260, 0.35, 'sawtooth', 0.09, 65); }
  cleared(): void { this.tone(420, 0.1, 'triangle', 0.055, 620); window.setTimeout(() => this.tone(620, 0.18, 'triangle', 0.055, 980), 105); }
  timeout(): void { this.tone(240, 0.16, 'sine', 0.05, 180); }

  private tone(start: number, duration: number, type: OscillatorType, gain: number, end: number): void {
    if (!this.enabled || !this.context || this.context.state !== 'running') return;
    const now = this.context.currentTime;
    const oscillator = this.context.createOscillator();
    const envelope = this.context.createGain();
    oscillator.type = type;
    oscillator.frequency.setValueAtTime(start, now);
    oscillator.frequency.exponentialRampToValueAtTime(Math.max(1, end), now + duration);
    envelope.gain.setValueAtTime(0.0001, now);
    envelope.gain.exponentialRampToValueAtTime(gain, now + 0.012);
    envelope.gain.exponentialRampToValueAtTime(0.0001, now + duration);
    oscillator.connect(envelope).connect(this.context.destination);
    oscillator.start(now);
    oscillator.stop(now + duration + 0.02);
  }
}
