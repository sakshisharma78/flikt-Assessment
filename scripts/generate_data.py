import csv
import random
from faker import Faker
import argparse


fake = Faker()


example_texts = [
'I love the product, it works perfectly and the support was great!',
'The delivery was late and the product arrived damaged. Very upset.',
'I have a question about installation, documentation is unclear.',
'Great value for money — satisfied with the purchase.',
'Support never replied to my email. Bad experience.',
'The mobile app crashes frequently on my phone.',
'Amazing features but pricing is confusing.',
]


if __name__ == '__main__':
parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=1500)
parser.add_argument('--out', type=str, default='data/sample_feedback.csv')
args = parser.parse_args()


sources = ['email', 'chat', 'social']
products = ['AppX', 'ServiceY', 'ProductZ', 'PlatformA']


with open(args.out, 'w', newline='', encoding='utf-8') as f:
writer = csv.writer(f)
writer.writerow(['id','source','date','customer_id','text','product','rating'])
for i in range(1, args.n+1):
text = random.choice(example_texts)
# add variations
if random.random() < 0.3:
text += ' ' + fake.sentence(nb_words=8)
writer.writerow([
i,
random.choice(sources),
fake.date_between(start_date='-1y', end_date='today').isoformat(),
fake.uuid4(),
text,
random.choice(products),
random.choice([None,1,2,3,4,5])
])


print(f'Generated {args.n} rows -> {args.out}')