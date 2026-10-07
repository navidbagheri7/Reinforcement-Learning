"""Readable rollout captions with a step counter, including stationary policies."""

from PIL import Image, ImageDraw


def caption_frames(frames, title, subtitle, stride, length):
    images = []
    for index, frame in enumerate(frames):
        image = Image.fromarray(frame)
        if image.width < 480:
            scale = 480 / image.width
            image = image.resize((480, int(image.height * scale)), Image.Resampling.NEAREST)
        canvas = Image.new("RGB", (image.width, image.height + 48), "#172637")
        canvas.paste(image, (0, 48))
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 7), title, fill="white")
        draw.text(
            (10, 26), f"{subtitle} | step {min(index * stride, length)}/{length}", fill="#c9e2f7"
        )
        images.append(canvas)
    return images
