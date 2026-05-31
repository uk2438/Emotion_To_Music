import argparse
import csv
from datetime import datetime
from pathlib import Path

from main import (
    EXPERIMENTS_ROOT,
    MODE_PROFILES,
    calculate_melody_metrics,
    calculate_selection_score,
    convert_to_note_names,
    export_melody_to_midi,
    generate_melody,
    recent_average,
    save_loss_csv,
    save_reward_breakdown_csv,
    save_reward_csv,
    set_seed,
    summarize_reward_breakdowns,
    train_agent,
    train_dqn_agent,
    train_factorized_dqn_agent,
)


DEFAULT_MOODS = ["happy", "sad", "angry", "neutral", "excited", "dark", "unstable"]


def _train_for_algorithm(args, mood, seed):
    common_kwargs = {
        "mode": mood,
        "episodes": args.episodes,
        "melody_length": args.melody_length,
        "mood_vector": {label: 1.0 if label == mood else 0.0 for label in DEFAULT_MOODS},
        "octave_expansion": args.octave_expansion,
        "expansion_start_ratio": args.expansion_start_ratio,
        "max_pitch_jump": args.max_pitch_jump,
        "action_masking": not args.disable_action_masking,
        "reward_weight_overrides": None,
    }

    if args.algorithm == "q_learning":
        env, agent, rewards = train_agent(
            state_mode="table",
            **common_kwargs,
        )
        return env, agent, rewards, None, "table"

    if args.algorithm == "dqn":
        env, agent, rewards, training_metrics = train_dqn_agent(
            batch_size=args.batch_size,
            target_update_interval=args.target_update_interval,
            hidden_size=args.dqn_hidden_size,
            learning_rate=args.dqn_learning_rate,
            epsilon_decay=args.dqn_epsilon_decay,
            epsilon_min=args.dqn_epsilon_min,
            replay_capacity=args.dqn_replay_capacity,
            use_double_dqn=not args.disable_double_dqn,
            **common_kwargs,
        )
        return env, agent, rewards, training_metrics, "vector"

    env, agent, rewards, training_metrics = train_factorized_dqn_agent(
        batch_size=args.batch_size,
        target_update_interval=args.target_update_interval,
        hidden_size=args.dqn_hidden_size,
        learning_rate=args.dqn_learning_rate,
        epsilon_decay=args.dqn_epsilon_decay,
        epsilon_min=args.dqn_epsilon_min,
        replay_capacity=args.dqn_replay_capacity,
        use_double_dqn=not args.disable_double_dqn,
        **common_kwargs,
    )
    return env, agent, rewards, training_metrics, "vector"


def _write_summary_csv(rows, filename):
    fieldnames = [
        "mood",
        "sample",
        "seed",
        "algorithm",
        "last_100_avg_reward",
        "best_training_reward",
        "best_selection_score",
        "generated_reward",
        "generated_selection_score",
        "melodic_quality_score",
        "mode_contour_alignment",
        "cadence_stability",
        "same_adjacent_ratio",
        "phrase_repeat_ratio",
        "pitch_range",
        "average_abs_pitch_interval",
        "total_duration",
        "average_duration",
        "short_note_ratio",
        "average_velocity",
        "velocity_range",
        "midi_file",
        "sample_dir",
    ]
    with open(filename, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_batch(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    moods = args.moods or DEFAULT_MOODS
    batch_dir = EXPERIMENTS_ROOT / f"{timestamp}_mood_batch_{args.algorithm}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for mood in moods:
        for sample_index in range(1, args.samples + 1):
            seed = args.seed + (1000 * DEFAULT_MOODS.index(mood)) + sample_index - 1
            set_seed(seed)
            sample_dir = batch_dir / f"{mood}_{sample_index:02d}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n=== {mood} sample {sample_index}/{args.samples} | seed {seed} ===")
            env, agent, rewards, training_metrics, state_mode = _train_for_algorithm(args, mood, seed)
            actions, melody, durations, velocities, pitch_actions, duration_actions, velocity_actions, events, reward_breakdowns = generate_melody(env, agent)
            note_names = convert_to_note_names(melody)
            metrics = calculate_melody_metrics(
                actions=actions,
                melody=melody,
                durations=durations,
                velocities=velocities,
                pitch_actions=pitch_actions,
                mode=mood,
                base_notes=env.base_notes,
            )
            reward_summary = summarize_reward_breakdowns(reward_breakdowns)
            generated_selection_score = calculate_selection_score(
                reward_summary["total_reward"],
                metrics,
                mode=mood,
            )

            midi_filename = sample_dir / f"generated_{mood}_{sample_index:02d}.mid"
            if not args.no_midi:
                export_melody_to_midi(
                    melody,
                    filename=midi_filename,
                    durations=durations,
                    velocities=velocities,
                    tempo=MODE_PROFILES[mood]["tempo"],
                    instrument_name=MODE_PROFILES[mood]["instrument"],
                    chord_progression=MODE_PROFILES[mood]["chord_progression"],
                )

            save_reward_csv(rewards, sample_dir / "episode_rewards.csv")
            if training_metrics and "episode_losses" in training_metrics:
                save_loss_csv(training_metrics["episode_losses"], sample_dir / "episode_losses.csv")
            save_reward_breakdown_csv(reward_breakdowns, sample_dir / "generated_reward_breakdown.csv")

            rows.append({
                "mood": mood,
                "sample": sample_index,
                "seed": seed,
                "algorithm": args.algorithm,
                "last_100_avg_reward": recent_average(rewards, 100),
                "best_training_reward": max(rewards) if rewards else "",
                "best_selection_score": (
                    training_metrics.get("best_selection_score", "")
                    if training_metrics else ""
                ),
                "generated_reward": reward_summary["total_reward"],
                "generated_selection_score": generated_selection_score,
                "melodic_quality_score": metrics["melodic_quality_score"],
                "mode_contour_alignment": metrics["mode_contour_alignment"],
                "cadence_stability": metrics["cadence_stability"],
                "same_adjacent_ratio": metrics["same_adjacent_ratio"],
                "phrase_repeat_ratio": metrics["phrase_repeat_ratio"],
                "pitch_range": metrics["pitch_range"],
                "average_abs_pitch_interval": metrics["average_abs_pitch_interval"],
                "total_duration": metrics["total_duration"],
                "average_duration": metrics["average_duration"],
                "short_note_ratio": metrics["short_note_ratio"],
                "average_velocity": metrics["average_velocity"],
                "velocity_range": metrics["velocity_range"],
                "midi_file": "" if args.no_midi else str(midi_filename),
                "sample_dir": str(sample_dir),
            })

            print("Note Names:", note_names)
            print("Metrics:", metrics)
            print("Generated reward:", round(reward_summary["total_reward"], 2))
            print("Generated selection score:", round(generated_selection_score, 2))

    summary_csv = batch_dir / "summary.csv"
    _write_summary_csv(rows, summary_csv)
    print("\nBatch summary saved:", summary_csv)
    return batch_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Generate comparable melodies for multiple moods.")
    parser.add_argument(
        "--algorithm",
        choices=["q_learning", "dqn", "factorized_dqn"],
        default="factorized_dqn",
    )
    parser.add_argument(
        "--moods",
        nargs="*",
        choices=DEFAULT_MOODS,
        help="Moods to generate. Defaults to all moods.",
    )
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--melody-length", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--target-update-interval", type=int, default=100)
    parser.add_argument("--dqn-hidden-size", type=int, default=128)
    parser.add_argument("--dqn-learning-rate", type=float, default=0.0005)
    parser.add_argument("--dqn-epsilon-decay", type=float, default=0.999)
    parser.add_argument("--dqn-epsilon-min", type=float, default=0.05)
    parser.add_argument("--dqn-replay-capacity", type=int, default=20000)
    parser.add_argument("--disable-double-dqn", action="store_true")
    parser.add_argument("--octave-expansion", action="store_true")
    parser.add_argument("--expansion-start-ratio", type=float, default=0.5)
    parser.add_argument("--max-pitch-jump", type=int, default=12)
    parser.add_argument("--disable-action-masking", action="store_true")
    parser.add_argument("--no-midi", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_batch(parse_args())
