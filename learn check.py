import random

class smorti:
    def __init__(self, inp, sped):
        self.input = inp
        self.weights = [random.uniform(-1,1) for _ in range(inp)]
        self.bias = random.uniform(-1,1)
        self.speed = sped
    def filter(self, num):
        return max(0, num)
    def passing(self, inputs):
        total = sum(i * w for i, w in zip(inputs, self.weights)) + self.bias
        return self.filter(total)
    def learn(self, inputs, target):
        output = self.passing(inputs)
        missing = target - output
        for i in range(len(self.weights)):
            self.weights[i] += self.speed * missing * inputs[i]
        self.bias += self.speed * missing

test = [
    smorti(3, .1),
    smorti(3, .1)
]
td = (
    ([0.2, -0.1, 0.6], 1),
    ([0.1, 0.2, -0.1], 0)
)

for step in range(5):
    for smort in test:
        print(f'\n--test {step + 1}--')
        for inputs, target in td:
            out = smort.learn(inputs, target)
            print(f"inputs: {inputs}, target: {target}, output: {out:.3f}")

for smort in test:
    print("\nFinal weights:", smort.weights)
    print("Final bias:", smort.bias)
