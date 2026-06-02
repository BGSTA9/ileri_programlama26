"""Titanic hayatta kalma tahmini - el yapimi MLP demonstrasyonu.

Algoritma akisi:
1. titanic.csv okunur.
2. Girdi olarak sadece uc anlamli ozellik secilir: Pclass, Sex, Age.
3. Eksik yas degerleri medyan ile doldurulur ve ozellikler 0-1 araligina olceklendirilir.
4. Agirliklar rastgele baslatilir.
5. Her epoch'ta tum ornekler icin forward pass yapilir.
6. Binary cross-entropy hata hesabi yapilir.
7. Backpropagation ile gradyanlar bulunur.
8. Agirliklar gradient descent ile guncellenir.
9. Ekranda epoch, loss, accuracy ve secili yolcu icin tahmin goruntulenir.
10. Ornek giris icin sayisal hesap aciklanir: sex=male, pclass=2, age=30 ise
    giris vektoru x = [0.5, 0.0, 30/max_age] olur; ornek olarak
    h1 = sigma(x1*w1 + x2*w2 + x3*w3 + b1)
    = sigma(0.5*0.5 + 0.0*(-0.2) + (30/max_age)*0.1 + 0.05)
    ve cikis icin
    y = sigma(h1*u1 + h2*u2 + ... + bo)
    seklinde hesap yapilir.
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path

try:
    import pygame
except ImportError as exc:
    raise SystemExit(
        "This demo requires pygame. Install it with: uv pip install pygame"
    ) from exc


WIDTH = 1600
HEIGHT = 940
FPS = 60

BG_TOP = (13, 17, 27)
BG_BOTTOM = (32, 40, 66)
PANEL = (18, 24, 38)
PANEL_EDGE = (84, 96, 126)
TEXT = (240, 244, 255)
MUTED = (163, 172, 201)
GREEN = (86, 225, 157)
RED = (255, 111, 126)
AMBER = (255, 196, 87)
BLUE = (82, 163, 255)
WHITE = (250, 250, 252)


@dataclass
class PassengerSample:
    passenger_id: int
    survived: int
    pclass: float
    sex: float
    age: float
    features: list[float]


@dataclass
class NetworkParams:
    hidden_weights: list[list[float]]
    hidden_biases: list[float]
    output_weights: list[float]
    output_bias: float


@dataclass
class EpochState:
    epoch: int
    train_loss: float
    train_accuracy: float
    validation_loss: float
    validation_accuracy: float
    sample: PassengerSample
    hidden_activations: list[float]
    output_probability: float
    train_history: list[float]
    accuracy_history: list[float]
    validation_history: list[float]


def lerp_color(color_a: tuple[int, int, int], color_b: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    amount = max(0.0, min(1.0, amount))
    return tuple(int(color_a[i] + (color_b[i] - color_a[i]) * amount) for i in range(3))


def make_gradient_background(size: tuple[int, int]) -> pygame.Surface:
    surface = pygame.Surface(size)
    height = size[1]
    for y in range(height):
        t = y / max(1, height - 1)
        color = lerp_color(BG_TOP, BG_BOTTOM, t)
        pygame.draw.line(surface, color, (0, y), (size[0], y))
    return surface


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def clamp_probability(value: float) -> float:
    return min(1.0 - 1e-7, max(1e-7, value))


def binary_cross_entropy(prediction: float, target: float) -> float:
    prediction = clamp_probability(prediction)
    return -(target * math.log(prediction) + (1.0 - target) * math.log(1.0 - prediction))


def accuracy_from_probability(prediction: float, target: int) -> float:
    predicted_label = 1 if prediction >= 0.5 else 0
    return 1.0 if predicted_label == target else 0.0


def load_titanic_samples(csv_path: Path) -> list[PassengerSample]:
    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows.extend(reader)

    ages = [float(row["Age"]) for row in rows if row.get("Age")]
    age_median = sorted(ages)[len(ages) // 2]
    max_age = max(max(ages), age_median)

    samples: list[PassengerSample] = []
    for row in rows:
        age_text = row.get("Age", "").strip()
        age = float(age_text) if age_text else age_median
        pclass = float(row["Pclass"])
        sex = 1.0 if row.get("Sex", "male").strip().lower() == "female" else 0.0
        survived = int(row["Survived"])
        passenger_id = int(row["PassengerId"])

        pclass_scaled = (3.0 - pclass) / 2.0
        age_scaled = min(age / max_age, 1.0)
        features = [pclass_scaled, sex, age_scaled]

        samples.append(
            PassengerSample(
                passenger_id=passenger_id,
                survived=survived,
                pclass=pclass,
                sex=sex,
                age=age,
                features=features,
            )
        )

    return samples


def split_samples(samples: list[PassengerSample], train_ratio: float = 0.8, seed: int = 19) -> tuple[list[PassengerSample], list[PassengerSample]]:
    shuffled = samples[:]
    random.Random(seed).shuffle(shuffled)
    split_index = max(1, int(len(shuffled) * train_ratio))
    return shuffled[:split_index], shuffled[split_index:]


def initialize_network(input_size: int, hidden_size: int, seed: int = 12) -> NetworkParams:
    rng = random.Random(seed)
    hidden_weights = [
        [rng.uniform(-0.9, 0.9) for _ in range(input_size)]
        for _ in range(hidden_size)
    ]
    hidden_biases = [rng.uniform(-0.4, 0.4) for _ in range(hidden_size)]
    output_weights = [rng.uniform(-0.9, 0.9) for _ in range(hidden_size)]
    output_bias = rng.uniform(-0.4, 0.4)
    return NetworkParams(hidden_weights, hidden_biases, output_weights, output_bias)


def forward_pass(features: list[float], params: NetworkParams) -> tuple[list[float], float]:
    hidden_activations: list[float] = []
    for node_index, node_weights in enumerate(params.hidden_weights):
        total = params.hidden_biases[node_index]
        for feature_value, weight in zip(features, node_weights):
            total += feature_value * weight
        hidden_activations.append(sigmoid(total))

    output_total = params.output_bias
    for hidden_value, weight in zip(hidden_activations, params.output_weights):
        output_total += hidden_value * weight
    output_probability = sigmoid(output_total)
    return hidden_activations, output_probability


def train_sample(sample: PassengerSample, params: NetworkParams, learning_rate: float) -> tuple[float, float]:
    hidden_activations, prediction = forward_pass(sample.features, params)
    target = float(sample.survived)
    loss = binary_cross_entropy(prediction, target)

    output_delta = prediction - target
    hidden_deltas: list[float] = []
    for hidden_index, hidden_activation in enumerate(hidden_activations):
        hidden_error = output_delta * params.output_weights[hidden_index]
        hidden_delta = hidden_error * hidden_activation * (1.0 - hidden_activation)
        hidden_deltas.append(hidden_delta)

    for hidden_index in range(len(params.output_weights)):
        params.output_weights[hidden_index] -= learning_rate * output_delta * hidden_activations[hidden_index]
    params.output_bias -= learning_rate * output_delta

    for hidden_index, hidden_delta in enumerate(hidden_deltas):
        for feature_index, feature_value in enumerate(sample.features):
            params.hidden_weights[hidden_index][feature_index] -= learning_rate * hidden_delta * feature_value
        params.hidden_biases[hidden_index] -= learning_rate * hidden_delta

    return loss, prediction


def evaluate(samples: list[PassengerSample], params: NetworkParams) -> tuple[float, float]:
    if not samples:
        return 0.0, 0.0

    total_loss = 0.0
    total_accuracy = 0.0
    for sample in samples:
        _, prediction = forward_pass(sample.features, params)
        total_loss += binary_cross_entropy(prediction, sample.survived)
        total_accuracy += accuracy_from_probability(prediction, sample.survived)

    count = len(samples)
    return total_loss / count, total_accuracy / count


def train_epoch(
    train_samples: list[PassengerSample],
    validation_samples: list[PassengerSample],
    params: NetworkParams,
    learning_rate: float,
    epoch: int,
) -> tuple[float, float, float, float, PassengerSample, list[float], float]:
    order = train_samples[:]
    random.Random(epoch + 31).shuffle(order)

    total_loss = 0.0
    total_accuracy = 0.0
    last_sample = order[0]
    last_hidden: list[float] = []
    last_prediction = 0.0

    for sample in order:
        loss, prediction = train_sample(sample, params, learning_rate)
        hidden_activations, last_prediction = forward_pass(sample.features, params)
        last_hidden = hidden_activations
        total_loss += loss
        total_accuracy += accuracy_from_probability(prediction, sample.survived)
        last_sample = sample

    train_loss = total_loss / len(order)
    train_accuracy = total_accuracy / len(order)
    validation_loss, validation_accuracy = evaluate(validation_samples, params)
    return train_loss, train_accuracy, validation_loss, validation_accuracy, last_sample, last_hidden, last_prediction


def compute_positions(layer_sizes: list[int], panel_rect: pygame.Rect) -> list[list[tuple[int, int]]]:
    x_start = panel_rect.left + 120
    x_end = panel_rect.right - 120
    if len(layer_sizes) == 1:
        x_positions = [panel_rect.centerx]
    else:
        step = (x_end - x_start) / max(1, len(layer_sizes) - 1)
        x_positions = [int(x_start + i * step) for i in range(len(layer_sizes))]

    layer_positions: list[list[tuple[int, int]]] = []
    top = panel_rect.top + 100
    bottom = panel_rect.bottom - 110

    for layer_index, count in enumerate(layer_sizes):
        if count == 1:
            y_positions = [panel_rect.centery]
        else:
            spacing = (bottom - top) / max(1, count - 1)
            y_positions = [int(top + i * spacing) for i in range(count)]
        layer_positions.append([(x_positions[layer_index], y) for y in y_positions])

    return layer_positions


def draw_panel(surface: pygame.Surface, rect: pygame.Rect, radius: int = 20) -> None:
    pygame.draw.rect(surface, PANEL, rect, border_radius=radius)
    pygame.draw.rect(surface, PANEL_EDGE, rect, width=2, border_radius=radius)


def draw_background(screen: pygame.Surface, background: pygame.Surface) -> None:
    screen.blit(background, (0, 0))
    accent = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    for index in range(12):
        alpha = 18 - index
        radius = 280 + index * 35
        pygame.draw.circle(accent, (255, 255, 255, max(0, alpha)), (WIDTH - 160, 140), radius, width=2)
    screen.blit(accent, (0, 0))


def draw_edge(screen: pygame.Surface, start: tuple[int, int], end: tuple[int, int], weight: float) -> None:
    intensity = min(1.0, abs(weight))
    thickness = max(1, int(1 + intensity * 5))
    base = BLUE if weight >= 0 else RED
    shade = lerp_color((42, 48, 70), base, intensity)
    pygame.draw.line(screen, shade, start, end, thickness)


def draw_network(
    screen: pygame.Surface,
    title_font: pygame.font.Font,
    small_font: pygame.font.Font,
    state: EpochState,
    params: NetworkParams,
    positions: list[list[tuple[int, int]]],
    panel_rect: pygame.Rect,
) -> None:
    input_values = state.sample.features

    for hidden_index, start in enumerate(positions[0]):
        for input_index, end in enumerate(positions[1]):
            draw_edge(screen, start, end, params.hidden_weights[input_index][hidden_index])

    for hidden_index, start in enumerate(positions[1]):
        draw_edge(screen, start, positions[2][0], params.output_weights[hidden_index])

    input_labels = ["Pclass", "Sex", "Age"]
    input_values_display = [
        f"{int(state.sample.pclass)}",
        "female" if state.sample.sex > 0.5 else "male",
        f"{state.sample.age:.1f}",
    ]

    for node_index, center in enumerate(positions[0]):
        activation = input_values[node_index]
        fill = lerp_color((30, 38, 56), GREEN, activation)
        pygame.draw.circle(screen, fill, center, 24)
        pygame.draw.circle(screen, WHITE, center, 24, 2)
        label = small_font.render(f"{input_labels[node_index]}", True, TEXT)
        value = small_font.render(f"{input_values_display[node_index]}", True, MUTED)
        screen.blit(label, label.get_rect(center=(center[0], center[1] - 42)))
        screen.blit(value, value.get_rect(center=(center[0], center[1] + 42)))

    for node_index, center in enumerate(positions[1]):
        activation = state.hidden_activations[node_index]
        fill = lerp_color((30, 38, 56), GREEN, activation)
        glow_color = lerp_color((62, 80, 116), WHITE, activation * 0.5)
        pygame.draw.circle(screen, glow_color, center, 26)
        pygame.draw.circle(screen, fill, center, 20)
        pygame.draw.circle(screen, WHITE, center, 20, 2)
        value = small_font.render(f"{activation:.2f}", True, TEXT)
        screen.blit(value, value.get_rect(center=(center[0], center[1] + 34)))

    output_center = positions[2][0]
    output_fill = lerp_color((30, 38, 56), GREEN, state.output_probability)
    pygame.draw.circle(screen, (72, 94, 136), output_center, 30)
    pygame.draw.circle(screen, output_fill, output_center, 22)
    pygame.draw.circle(screen, WHITE, output_center, 22, 2)

    output_label = title_font.render("Survival", True, TEXT)
    probability_label = small_font.render(f"p={state.output_probability:.2f}", True, AMBER)
    screen.blit(output_label, output_label.get_rect(center=(output_center[0], output_center[1] - 48)))
    screen.blit(probability_label, probability_label.get_rect(center=(output_center[0], output_center[1] + 42)))

    for layer_index, layer_name in enumerate(["Input", "Hidden", "Output"]):
        label = title_font.render(layer_name, True, TEXT)
        center_x = positions[layer_index][0][0]
        screen.blit(label, label.get_rect(center=(center_x, panel_rect.top + 42)))


def draw_metrics_panel(
    screen: pygame.Surface,
    title_font: pygame.font.Font,
    small_font: pygame.font.Font,
    state: EpochState,
    rect: pygame.Rect,
) -> None:
    draw_panel(screen, rect)

    title = title_font.render("Epoch Metrics", True, TEXT)
    screen.blit(title, (rect.left + 22, rect.top + 18))

    metrics = [
        ("Epoch", f"{state.epoch}"),
        ("Train loss", f"{state.train_loss:.4f}"),
        ("Train acc", f"{state.train_accuracy * 100:.1f}%"),
        ("Val loss", f"{state.validation_loss:.4f}"),
        ("Val acc", f"{state.validation_accuracy * 100:.1f}%"),
        ("Learning rate", "0.08"),
    ]

    start_y = rect.top + 72
    for index, (label_text, value_text) in enumerate(metrics):
        label = small_font.render(label_text, True, MUTED)
        value_color = AMBER if "loss" in label_text.lower() else GREEN if "acc" in label_text.lower() else TEXT
        value = small_font.render(value_text, True, value_color)
        y = start_y + index * 34
        screen.blit(label, (rect.left + 22, y))
        screen.blit(value, (rect.right - 22 - value.get_width(), y))

    hint_lines = [
        "SPACE: pause / resume",
        "LEFT/RIGHT: epoch step",
        "R: reinitialize weights",
        "ESC: quit",
    ]
    hint_y = rect.bottom - 150
    for index, text in enumerate(hint_lines):
        hint = small_font.render(text, True, MUTED)
        screen.blit(hint, (rect.left + 22, hint_y + index * 28))


def draw_sample_panel(
    screen: pygame.Surface,
    title_font: pygame.font.Font,
    small_font: pygame.font.Font,
    state: EpochState,
    rect: pygame.Rect,
) -> None:
    draw_panel(screen, rect)

    title = title_font.render("Sample Snapshot", True, TEXT)
    screen.blit(title, (rect.left + 22, rect.top + 18))

    sample = state.sample
    lines = [
        f"Passenger ID: {sample.passenger_id}",
        f"Pclass: {int(sample.pclass)}",
        f"Sex: {'female' if sample.sex > 0.5 else 'male'}",
        f"Age: {sample.age:.1f}",
        f"Target survived: {sample.survived}",
        f"Prediction: {state.output_probability:.3f}",
    ]

    start_y = rect.top + 72
    for index, text in enumerate(lines):
        label = small_font.render(text, True, TEXT if index < 4 else AMBER)
        screen.blit(label, (rect.left + 22, start_y + index * 32))

    bar_x = rect.left + 22
    bar_y = rect.bottom - 74
    bar_width = rect.width - 44
    pygame.draw.rect(screen, (30, 38, 56), (bar_x, bar_y, bar_width, 18), border_radius=9)
    fill_width = int(bar_width * state.output_probability)
    pygame.draw.rect(screen, GREEN, (bar_x, bar_y, fill_width, 18), border_radius=9)

    pred_label = small_font.render("Survival probability", True, MUTED)
    screen.blit(pred_label, (bar_x, bar_y - 24))


def draw_history_chart(
    screen: pygame.Surface,
    title_font: pygame.font.Font,
    small_font: pygame.font.Font,
    state: EpochState,
    rect: pygame.Rect,
) -> None:
    draw_panel(screen, rect)

    title = title_font.render("Learning Curves", True, TEXT)
    screen.blit(title, (rect.left + 22, rect.top + 16))

    chart_rect = pygame.Rect(rect.left + 24, rect.top + 60, rect.width - 48, rect.height  )
    pygame.draw.rect(screen, (22, 29, 45), chart_rect, border_radius=16)

    if len(state.train_history) < 2:
        text = small_font.render("Epoch history will appear here.", True, MUTED)
        screen.blit(text, text.get_rect(center=chart_rect.center))
        return

    max_loss = max(max(state.train_history), max(state.validation_history), 0.25)
    min_loss = 0.0
    chart_padding = 10
    inner = pygame.Rect(
        chart_rect.left + chart_padding,
        chart_rect.top + chart_padding,
        chart_rect.width - chart_padding ,
        chart_rect.height  ,
    )

    def loss_point(index: int, value: float) -> tuple[int, int]:
        x = inner.left + int(index / max(1, len(state.train_history) - 1) * inner.width)
        normalized = (value - min_loss) / max(1e-7, max_loss - min_loss)
        y = inner.bottom - int(normalized * inner.height)
        return x, y

    def acc_point(index: int, value: float) -> tuple[int, int]:
        x = inner.left + int(index / max(1, len(state.accuracy_history) - 1) * inner.width)
        y = inner.bottom - int(value * inner.height)
        return x, y

    for index in range(0, 5):
        y = inner.top + int(index / 4 * inner.height)
        pygame.draw.line(screen, (42, 49, 70), (inner.left, y), (inner.right, y), 1)

    train_points = [loss_point(i, value) for i, value in enumerate(state.train_history)]
    validation_points = [loss_point(i, value) for i, value in enumerate(state.validation_history)]
    accuracy_points = [acc_point(i, value) for i, value in enumerate(state.accuracy_history)]

    pygame.draw.lines(screen, AMBER, False, train_points, 3)
    pygame.draw.lines(screen, RED, False, validation_points, 3)
    pygame.draw.lines(screen, GREEN, False, accuracy_points, 3)

    legend = [
        (AMBER, "Train loss"),
        (RED, "Val loss"),
        (GREEN, "Train acc"),
    ]
    legend_x = rect.left + 22
    legend_y = rect.bottom - 24
    for index, (color, text_value) in enumerate(legend):
        pygame.draw.rect(screen, color, (legend_x + index * 150, legend_y, 12, 12))
        label = small_font.render(text_value, True, MUTED)
        screen.blit(label, (legend_x + index * 150 + 18, legend_y - 3))


def main() -> None:
    pygame.init()
    pygame.display.set_caption("Titanic Survival MLP")

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    title_font = pygame.font.Font(None, 30)
    small_font = pygame.font.Font(None, 22)

    background = make_gradient_background((WIDTH, HEIGHT))

    dataset_path = Path(__file__).resolve().parents[1] / "titanic.csv"
    samples = load_titanic_samples(dataset_path)
    train_samples, validation_samples = split_samples(samples)

    params = initialize_network(input_size=3, hidden_size=5)
    learning_rate = 0.08
    max_epochs = 80
    current_epoch = 0
    paused = False
    accumulated_frame = 0
    train_history: list[float] = []
    accuracy_history: list[float] = []
    validation_history: list[float] = []

    network_rect = pygame.Rect(40, 40, 920, 740)
    metrics_rect = pygame.Rect(1000, 40, 560, 220)
    sample_rect = pygame.Rect(1000, 280, 560, 220)
    chart_rect = pygame.Rect(1000, 520, 560, 260)

    layer_positions = compute_positions([3, 5, 1], network_rect)
    highlighted_sample = train_samples[0]
    highlighted_hidden, highlighted_prediction = forward_pass(highlighted_sample.features, params)
    state = EpochState(
        epoch=0,
        train_loss=0.0,
        train_accuracy=0.0,
        validation_loss=0.0,
        validation_accuracy=0.0,
        sample=highlighted_sample,
        hidden_activations=highlighted_hidden,
        output_probability=highlighted_prediction,
        train_history=train_history,
        accuracy_history=accuracy_history,
        validation_history=validation_history,
    )

    running = True
    print("SPACE: pause/resume, LEFT/RIGHT: epoch step, R: reset, ESC: quit")

    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_RIGHT:
                    paused = True
                    current_epoch = min(max_epochs, current_epoch + 1)
                elif event.key == pygame.K_LEFT:
                    paused = True
                    current_epoch = max(0, current_epoch - 1)
                elif event.key == pygame.K_r:
                    params = initialize_network(input_size=3, hidden_size=5, seed=random.randint(1, 10_000))
                    current_epoch = 0
                    paused = False
                    accumulated_frame = 0
                    train_history.clear()
                    accuracy_history.clear()
                    validation_history.clear()

        if not paused:
            accumulated_frame += 1
            if accumulated_frame >= 4:
                accumulated_frame = 0
                if current_epoch < max_epochs:
                    current_epoch += 1
                else:
                    current_epoch = 0
                    params = initialize_network(input_size=3, hidden_size=5, seed=random.randint(1, 10_000))
                    train_history.clear()
                    accuracy_history.clear()
                    validation_history.clear()

        if len(train_history) != current_epoch + 1 or current_epoch == 0 and not train_history:
            train_loss, train_accuracy, validation_loss, validation_accuracy, sample, hidden_activations, output_probability = train_epoch(
                train_samples=train_samples,
                validation_samples=validation_samples,
                params=params,
                learning_rate=learning_rate,
                epoch=current_epoch,
            )

            if current_epoch == 0 and not train_history:
                train_history.append(train_loss)
                accuracy_history.append(train_accuracy)
                validation_history.append(validation_loss)
            else:
                while len(train_history) < current_epoch:
                    train_history.append(train_history[-1])
                    accuracy_history.append(accuracy_history[-1])
                    validation_history.append(validation_history[-1])
                if len(train_history) == current_epoch:
                    train_history.append(train_loss)
                    accuracy_history.append(train_accuracy)
                    validation_history.append(validation_loss)
                else:
                    train_history[current_epoch] = train_loss
                    accuracy_history[current_epoch] = train_accuracy
                    validation_history[current_epoch] = validation_loss

            highlighted_sample = sample
            highlighted_hidden = hidden_activations
            highlighted_prediction = output_probability
            state = EpochState(
                epoch=current_epoch,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                validation_loss=validation_loss,
                validation_accuracy=validation_accuracy,
                sample=highlighted_sample,
                hidden_activations=highlighted_hidden,
                output_probability=highlighted_prediction,
                train_history=train_history[:],
                accuracy_history=accuracy_history[:],
                validation_history=validation_history[:],
            )

        draw_background(screen, background)

        draw_panel(screen, network_rect)
        network_title = title_font.render("Titanic Survival MLP", True, TEXT)
        screen.blit(network_title, (network_rect.left + 22, network_rect.top + 16))
        subtitle = small_font.render(
            "3 inputs: Pclass, Sex, Age | 1 hidden layer | 1 survival output",
            True,
            MUTED,
        )
        screen.blit(subtitle, (network_rect.left + 22, network_rect.top + 48))
        draw_network(screen, title_font, small_font, state, params, layer_positions, network_rect)

        draw_metrics_panel(screen, title_font, small_font, state, metrics_rect)
        draw_sample_panel(screen, title_font, small_font, state, sample_rect)
        draw_history_chart(screen, title_font, small_font, state, chart_rect)

        footer = small_font.render(
            f"Mode: {'Paused' if paused else 'Running'}   |   Epoch {state.epoch}/{max_epochs}",
            True,
            WHITE,
        )
        screen.blit(footer, (40, HEIGHT - 28))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()