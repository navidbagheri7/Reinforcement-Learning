"""Animate existing finite navigation policies on the first fixed test episode."""

from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw

from decision_lab.artifacts import sha256, write_json
from decision_lab.deployment import load_policy


def maze_frame(env):
    cell = 40
    canvas = Image.new("RGB", (len(env.layout[0]) * cell, len(env.layout) * cell), "#f3f5f7")
    draw = ImageDraw.Draw(canvas)
    for row in range(len(env.layout)):
        for col in range(len(env.layout[0])):
            box = (col * cell, row * cell, (col + 1) * cell, (row + 1) * cell)
            color = "#293749" if env.layout[row][col] == "#" else "#e9edf2"
            draw.rectangle(box, fill=color, outline="white")
    row, col = env.positions[env.goal]
    draw.rectangle(
        (col * cell + 4, row * cell + 4, (col + 1) * cell - 4, (row + 1) * cell - 4), fill="#56b870"
    )
    row, col = env.positions[env.state]
    draw.ellipse(
        (col * cell + 8, row * cell + 8, (col + 1) * cell - 8, (row + 1) * cell - 8), fill="#327cc5"
    )
    return canvas


def main():
    destination = Path("assets/rollouts")
    for name in ["Maze", "FrozenLake-v1", "Taxi-v3"]:
        file = Path("assets/policies") / f"tabular_{name}_q_learning_0.json"
        metadata, make, policy = load_policy(file)
        env = make() if name == "Maze" else gym.make(name, render_mode="rgb_array")
        observation, _ = env.reset(seed=300000)
        frames, total, steps = [], 0.0, 0
        while True:
            frames.append(maze_frame(env) if name == "Maze" else Image.fromarray(env.render()))
            observation, reward, terminated, truncated, _ = env.step(int(policy[observation]))
            total += reward
            steps += 1
            if terminated or truncated:
                frames.append(maze_frame(env) if name == "Maze" else Image.fromarray(env.render()))
                break
        env.close()
        evidence = np.load(f"assets/source_data/tabular/{name}_q_learning_0.npz")
        assert np.isclose(total, evidence["returns"][0], rtol=0, atol=1e-12)
        assert steps == int(evidence["lengths"][0])
        gif = destination / f"{name}_q_learning_0.gif"
        frames[0].save(gif, save_all=True, append_images=frames[1:], duration=140, loop=0)
        write_json(
            gif.with_suffix(".json"),
            dict(
                env=name,
                algorithm="Q-learning",
                fit_seed=0,
                test_seed=300000,
                return_value=total,
                length=steps,
                terminated=terminated,
                truncated=truncated,
                gif_sha256=sha256(gif),
                policy_sha256=sha256(file),
                source_evidence=f"assets/source_data/tabular/{name}_q_learning_0.npz",
                protocol="First held-out episode, verified against saved returns and lengths; no visual selection",
                renderer="Owned 2D maze renderer"
                if name == "Maze"
                else "Official Gymnasium RGB renderer",
            ),
        )
        print(name, total, steps, gif)


if __name__ == "__main__":
    main()
