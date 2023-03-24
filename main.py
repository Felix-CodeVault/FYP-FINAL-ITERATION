import time, random, requests
from flask import Flask, render_template, request, session, redirect, url_for, jsonify
from flask_socketio import join_room, leave_room, send, emit, SocketIO
from string import ascii_uppercase
import numpy as np
import torch
from torch import nn
from PIL import Image
import re

app = Flask(__name__)
app.config["SECRET_KEY"] = "sdasd"
socketio = SocketIO(app)

rooms = {}

class_names_path = "static/class_names.txt"
torch_model_path = "static/pytorch_model.bin"

LABELS = open(class_names_path).read().splitlines()

model = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding='same'),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding='same'),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(64, 128, 3, padding='same'),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(1152, 256),
    nn.ReLU(),
    nn.Linear(256, len(LABELS)),
)
state_dict = torch.load(torch_model_path, map_location='cpu')
model.load_state_dict(state_dict, strict=False)
model.eval()


def predict(im):
    x = torch.tensor(im, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.

    with torch.no_grad():
        out = model(x)

    probabilities = torch.nn.functional.softmax(out[0], dim=0)

    values, indices = torch.topk(probabilities, 10)

    return {LABELS[i]: v.item() for i, v in zip(indices, values)}


def generate_unique_code(length):
    while True:
        code = ""
        for _ in range(length):
            code += random.choice(ascii_uppercase)

        if code not in rooms:
            break
    return code


@app.route("/game")
def game():
    return render_template("game_room.html")


@app.route("/", methods=["POST", "GET"])
def login():
    session.clear()
    if request.method == "POST":
        name = request.form.get("name")

        if not name:
            return render_template("login.html", error="Enter a Name", name=name)

        session["name"] = name
        print("User Created: ", name)
        return redirect(url_for("home"))

    return render_template("login.html")


@app.route("/home", methods=["POST", "GET"])
def home():
    name = session["name"]
    if request.method == "POST":
        code = request.form.get("code")
        join = request.form.get("join", False)
        create = request.form.get("create", False)

        # for joining a room
        if join != False and not code:
            render_template("home.html", error="Enter room code", code=code, name=name)

        room = code
        if create != False:
            room = generate_unique_code(5)
            rooms[room] = {"members": 0, "messages": []}
        elif code not in rooms:
            return render_template("home.html", error="Room doesn't exist", code=code, name=name)

        if rooms[room]["members"] >= 2:
            return render_template("home.html", error="Room at Max Capacity", code=code, name=name)

        session["room"] = room
        print(room)
        return redirect(url_for("prep_zone"))

    return render_template("home.html", name=name)


@app.route("/prep_zone", methods=["POST", "GET"])
def prep_zone():
    room = session.get("room")
    if room is None or session.get("name") is None or room not in rooms:
        return redirect(url_for("home"))

    if request.method == "POST":
        play = request.form.get("play-btn", False)
        if play != False:
            print(rooms[room]["members"])
            if rooms[room]["members"] == 1:
                return redirect(url_for("room"))
            else:
                return render_template("prep_zone.html", code=room, error="Second Player Needed",
                                       messages=rooms[room]["messages"])
    return render_template("prep_zone.html", code=room, messages=rooms[room]["messages"])


@app.route("/room")
def room():
    room = session.get("room")
    # starting the game when both players enter the room
    if rooms[room]["members"] == 2:
        return render_template("room.html", messages=rooms[room]["messages"])
    return render_template("room.html", messages=rooms[room]["messages"])


@app.route("/winner_screen")
def winner_screen():
    winner = request.args.get("winnerValue")
    return render_template("winnerScreen.html", winner=winner)


@socketio.on("message")
def message(data):
    room = session.get("room")
    if room not in rooms:
        return

    content = {
        "name": session.get("name"),
        "message": data["data"]
    }
    send(content, to=room)
    rooms[room]["messages"].append(content)
    print(f'{session.get("name")} sent: {data["data"]}')


@socketio.on("connect")
def connect(auth):
    room = session.get("room")
    name = session.get("name")
    if not room or not name:
        return
    if room not in rooms:
        leave_room(room)
        return

    # join room
    join_room(room)
    send({"name": name, "message": "has entered the room"}, to=room)
    rooms[room]["members"] += 1
    print(f"{name} connected to room {room}")


@socketio.on("disconnect")
def disconnect():
    room = session.get("room")
    name = session.get("name")
    # allows client to rejoin on reload
    # time.sleep(1)
    leave_room(room)

    if room in rooms:
        rooms[room]["members"] -= 1
        # waiting before checking if room can be deleted
        # this ensures ample time to reconnect on page reload
        time.sleep(5)
        if rooms[room]["members"] <= 0:
            del rooms[room]

    send({"name": name, "message": "has left the room"}, to=room)
    print(f"{name} disconnected from room {room}")


@socketio.on("canvas_data")
def handle_canvas_data(data):
    # TODO: deals with drawing recognition
    # send data to all canvases
    emit("canvas_data", data, broadcast=True, include_self=False)


@socketio.on("canvas_data_player_2")
def handle_canvas_data_player_2(data):
    # TODO: deals with drawing recognition
    # send data to all canvases
    emit("canvas_data_player_2", data, broadcast=True, include_self=False)


@socketio.on("canvas_data_array")
def handle_guess(canvas_data_array):
    guess1 = recognise_image(canvas_data_array)
    emit("guess-player-1", guess1[0])


@socketio.on("canvas_data_array_player_2")
def handle_guess_player2(canvas_data_array2):
    guess2 = recognise_image(canvas_data_array2)
    emit("guess-player-2", guess2[0])

@socketio.on("all_data")
def handle_all_data(data):
    socketio.emit("data", data)

def recognise_image(data):
    flat_data = [pixel for row in data for pixel in row]  # convert to a flat list
    flat_data = np.array(flat_data, dtype=np.uint8)  # convert to numpy array of uint8 type

    # create a PIL image object from the numpy array
    img = Image.fromarray(flat_data.reshape(400, 400), mode='L')

    # resize the image to 28x28 using Lanczos filter
    img_resized = img.resize((28, 28), resample=Image.LANCZOS)

    # convert the resized image back to a numpy array
    resized_data = np.array(img_resized.getdata(), dtype=np.uint8).reshape(28, 28)

    # # Loop through each element in the 2D array
    # for i in range(len(resized_data)):
    #     for j in range(len(resized_data[i])):
    #         # Check if the element is "on" (equal to zero)
    #         if resized_data[i][j] != 0:
    #             # Convert the "on" value to 255
    #             resized_data[i][j] = 255

    # print(resized_data)

    guess = predict(resized_data)
    output = list(guess.keys())
    print(output)
    return output


if __name__ == "__main__":
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)
