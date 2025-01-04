import random

R_EATING = "I don't like eating anything because I'm a bot :)"

def unknown():
  """
  This function returns a default response for unknown inputs.
  """
  response = ["Could you please re-phrase that? ",
                "...",
                "Sounds about right.",
                "What does that mean?"][random.randrange(4)]
  return response