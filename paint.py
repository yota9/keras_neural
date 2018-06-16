import re
rxp = re.compile(r'/step - loss: (\d*\.\d+) - acc: (\d*\.\d+) - val_loss: (\d*\.\d+) - val_acc: (\d*\.\d+)')
losses, accs, test_losses, test_accs = [], [], [], []

with open('script.o3507') as fp:
    for line in fp.read().split('\n'):
        matches = rxp.search(line)
        
        if matches is None:
            continue
        
        loss, acc, test_loss, test_acc = matches.group(1), matches.group(2), matches.group(3), matches.group(4)
        
        losses.append(float(loss))
        accs.append(float(acc))
        test_losses.append(float(test_loss))
        test_accs.append(float(test_acc))
        
add_epoch_number = lambda iterable: [(epoch_num, x) for epoch_num, x in enumerate(iterable)]

losses = add_epoch_number(losses)
accs = add_epoch_number(accs)
test_losses = add_epoch_number(test_losses)
test_accs = add_epoch_number(test_accs)

import matplotlib.pyplot as plt

import seaborn as sns
sns.set()

assert len(losses) == len(accs) == len(test_losses) == len(test_accs)

x = [x for x in range(len(losses))]

fig = plt.figure()

ax = fig.add_subplot(111)
ax.set_xlabel("Номер эпохи")
ax.set_ylabel("Train loss")
ax.plot(x, [x[1] for x in losses])
plt.savefig('foo1.png')

fig = plt.figure()

ax = fig.add_subplot(111)
ax.set_xlabel("Номер эпохи")
ax.set_ylabel("Train acc")
ax.plot(x, [x[1] for x in accs])
plt.savefig('foo2.png')

from operator import itemgetter
max_acc_epoch, max_acc = max(accs, key=itemgetter(1))
min_loss_epoch, min_loss = min(losses, key=itemgetter(1))

print('Max acc on test is {:.2%} epoch {}'.format(max_acc, max_acc_epoch + 1))
print('Min loss on test is {} epoch {}'.format(min_loss, min_loss_epoch + 1))

print('Last acc on test {:.2%}'.format(test_accs[-1][1]))
