from trie import Trie 

#调用来自：        self.gaz_lower = False    self.gaz = Gazetteer(self.gaz_lower)
class Gazetteer:
    def __init__(self, lower): #lower=gaz_lower=False
        self.trie = Trie()
        self.ent2type = {} ## word list to type
        self.ent2id = {"<UNK>":0}   ## word list to id
        self.lower = lower
        self.space = ""       #包括gaz_file的所有词

	#调用自：data.gaz.enumerateMatchList(word_list[idx:])
    def enumerateMatchList(self, word_list):
        if self.lower:
            word_list = [word.lower() for word in word_list]
        match_list = self.trie.enumerateMatch(word_list, self.space)
        return match_list   #返回word_list在space中匹配到的单词列表

	#调用自 data.gaz.insert(fin, "one_source")   fin是gaz_file每行 中的词
    def insert(self, word_list, source):
        if self.lower:
            word_list = [word.lower() for word in word_list]
        self.trie.insert(word_list)
        string = self.space.join(word_list)
        if string not in self.ent2type:
            self.ent2type[string] = source
        if string not in self.ent2id:
            self.ent2id[string] = len(self.ent2id)

    def searchId(self, word_list):
        if self.lower:
            word_list = [word.lower() for word in word_list]
        string = self.space.join(word_list)
        if string in self.ent2id:
            return self.ent2id[string]
        return self.ent2id["<UNK>"]

    def searchType(self, word_list):
        if self.lower:
            word_list = [word.lower() for word in word_list]
        string = self.space.join(word_list)
        if string in self.ent2type:
            return self.ent2type[string]
        print  "Error in finding entity type at gazetteer.py, exit program! String:", string
        exit(0)

    def size(self):
        return len(self.ent2type)




