def word_ids(file):

 text_array = []
 d = {}
# d['SOS'] = len(d)+0
 i=1
#text_file = open('/home/srallaba/projects/personality_stuff/voices/cmu_us_LJspeech/Speech_Expts_Barebones/taco_25October_expts/Data/cmu_us_bdl_arctic/etc/new_txt.done.data')
 text_file= open(file).readlines()
 for line in text_file:
  line=line.strip()
#  print("line is", line)
  wordseq = line.split('\n')[0]
  words = wordseq.split(' ')
#  print("words are:", words)
  for l in range(0,len(words)): 
   if words[l] not in d.keys():
    d[words[l]] = i
    i = i+1
# print("d is", d['SOS'])
 return d
#print(d['slightly'])


def word_iids(words, d):
  i=1
  for l in range(0,len(words)): 
   if words[l] not in d.keys():
    d[words[l]] = i
    i = i+1
#  print("wordids are:", d)
  return d



def text_seq(text):
  wordseq = text.split('\n')[0]
  print("wordseq is", wordseq)
  words = wordseq.split(' ')
  word_id = word_iids(words, d)
#  print("word_id is", word_id)
  print("ids are", [word_id[word] for word in words])
  return [word_id[word] for word in words]
#  print("wordids are:", d)

