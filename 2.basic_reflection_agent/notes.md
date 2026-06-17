My reflect node was not generating output after second try, because the state had a lot of things and llm couldnt not figure out what to critique on

thats why->
I sent only last AI message(last tweet) to reflect node
but,
the langgraph still passes whole state to reflect node, but I am explicitly passing last tweet to llm

### What MessageGraph does internally

Conceptually:
state = []
state += [HumanMessage(...)] # initial input
state += [generate_node(state)]
state += [reflect_node(state)]
state += [generate_node(state)]
state += [reflect_node(state)]

### also, I have added langsmith see a clean flow there, checkout smith.langsmith.com

-> just added details in .env, @traceable wasnt really needed so far
