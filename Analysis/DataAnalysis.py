import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

multi_lang_df = pd.read_csv('/mnt/data/helpdesk_customer_multi_lang_tickets.csv')
helpdesk_df = pd.read_csv('/mnt/data/helpdesk_customer_tickets.csv')
support_df = pd.read_csv('/mnt/data/customer_support_tickets.csv')

language_distribution = multi_lang_df['language'].value_counts()
plt.figure(figsize=(8, 6))
sns.barplot(x=language_distribution.index, y=language_distribution.values)
plt.title('Ticket Distribution by Language')
plt.xlabel('Language')
plt.ylabel('Number of Tickets')
plt.savefig('language_distribution.png')
plt.show()

priority_distribution = helpdesk_df['priority'].value_counts()
plt.figure(figsize=(8, 6))
sns.barplot(x=priority_distribution.index, y=priority_distribution.values, palette='viridis')
plt.title('Ticket Priority Distribution')
plt.xlabel('Priority')
plt.ylabel('Number of Tickets')
plt.savefig('priority_distribution.png')
plt.show()

combined_df = pd.concat([multi_lang_df[['id', 'subject', 'language', 'priority']], 
                         helpdesk_df[['id', 'subject', 'language', 'priority']]], ignore_index=True)

plt.figure(figsize=(10, 8))
sns.countplot(data=combined_df, x='language', hue='priority', palette='coolwarm')
plt.title('Priority Distribution by Language')
plt.xlabel('Language')
plt.ylabel('Number of Tickets')
plt.legend(title='Priority')
plt.savefig('priority_by_language.png')
plt.show()

summary = {
    'total_tickets': combined_df.shape[0],
    'languages': combined_df['language'].unique().tolist(),
    'priorities': combined_df['priority'].unique().tolist()
}

with open('analysis_summary.txt', 'w') as f:
    f.write(str(summary))

print("Data analysis completed and saved.")
