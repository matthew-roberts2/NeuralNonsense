import discord
from net import load_names, build_model, generate
import tensorflow as tf
import numpy as np
import random
import re

TOKEN = 'NTYzMjA4NTA3NDcwNjQzMjIw.XKV--A.rsOuJ68ZlhVrBqYVmqyAFhE5twQ'

client = discord.Client()


def init_name_gen_model():
    names = load_names()
    vocab = [' '] + sorted(set(''.join([name for name in names])))
    vocab_size = len(vocab)
    embedding_dim = 256
    rnn_units = 16
    checkpoint_dir = './model/namegen'

    char_index = {u: i for i, u in enumerate(vocab)}
    index_char = np.array(vocab)

    model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    model.build(tf.TensorShape([1, None]))

    return model, index_char, char_index


name_gen_model, ngm_idx, ngm_char = init_name_gen_model()

temp = 1.0

intros = [
    "Hey-o, it's {}!",
    "Oh shit! It's {}!",
    "Looks like the cat dragged {} in tonight!",
    "What kind of name is {}, anyway?",
    "The only name worse than {} is Redeject!",
    "Give it up for {}!",
    "{} is already Tracer!",
    "Don't name your dog {}",
    "Who invited {}??",
    "{} would be a great name in my new hit single!",
]

use_single_intro = None


@client.event
async def on_message(message):
    global temp, use_single_intro, name_gen_model, ngm_char, ngm_idx

    if message.author == client.user:
        return

    if message.content.lower() == 'bingo reload':
        if str(message.author) != 'Mattpenguin#8371':
            await message.channel.send(
                f'I\'m sorry, {message.author.nick if isinstance(message.author, discord.member.Member) else message.author.name}, I\'m afraid I can\'t let you do that')
        else:
            async with message.channel.typing():
                name_gen_model, ngm_idx, ngm_char = init_name_gen_model()
                await message.channel.send('I feel like a whole new bot!')

    msg = message.content
    if 'Bingo' in msg and 'name' in msg.lower():
        async with message.channel.typing():
            if 'starting with' in msg.lower():
                index = ord((msg.lower())[-1]) - 96
            else:
                index = random.randint(1, 26)
            name = generate(name_gen_model, ngm_idx[index], ngm_char, ngm_idx, 50, temp).strip()
            bot_msg = intros[
                use_single_intro if use_single_intro is not None else random.randint(0, len(intros) - 1)].format(name)
            await message.channel.send(bot_msg)
        return

    if 'Bingo' in msg and 'set wackiness' in msg.lower():
        new_temp = float(re.search(r"(\d+\.?\d*)", msg).group(1))
        if new_temp < 0.5:
            bot_msg = "That's just not wacky enough! (Choose a floating point number in [0.5, 2])"
        elif new_temp > 2:
            bot_msg = "Whoa there, let's not get *too* crazy now! (Choose a floating point number in [0.5, 2])"
        else:
            temp = new_temp
            bot_msg = "Sounds like a plan!"
        await message.channel.send(bot_msg)
        return

    if 'Bingo' in msg and 'how wacky are you' in msg.lower():
        bot_msg = "On a scale of 0.5 to 2? A solid {}".format(temp)
        await message.channel.send(bot_msg)
        return

    if 'thanks bingo' in msg.lower():
        bot_msg = "No problemo!"
        await message.channel.send(bot_msg)
        return

    if 'Bingo' in msg and 'lets keep it simple' in msg.lower():
        use_single_intro = 0
        await message.channel.send("Yes, sir!")
        return

    if 'Bingo' in msg and 'give me some variety' in msg.lower():
        use_single_intro = None
        await message.channel.send("Oh yeah!")
        return


@client.event
async def on_ready():
    print('Logged in as')
    print(client.user.name)
    print(client.user.id)
    print('------')


client.run(TOKEN)
