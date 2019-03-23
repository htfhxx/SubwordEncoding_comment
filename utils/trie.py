#coding=utf-8
import collections

class TrieNode:
	
    # Initialize your data structure here.
    def __init__(self):
        self.children = collections.defaultdict(TrieNode) #构建一个类似dictionary的对象,其values类型为TrieNode
        self.is_word = False
		
#相当于一个节点，保存了无数个子节点children，以及一个is_word，
#每个子节点children都是TrieNode类型，children[letter]中的letter是加入的word
class Trie:
    def __init__(self):
        self.root = TrieNode()

	#调用自：data.gaz.trie.insert(word_list)
    def insert(self, word):
        
        current = self.root
        for letter in word:
            current = current.children[letter]
        current.is_word = True

    def search(self, word):
        current = self.root
        for letter in word:
            current = current.children.get(letter)
            if current is None:
                return False
        return current.is_word

    def startsWith(self, prefix):
        current = self.root
        for letter in prefix:
            current = current.children.get(letter)
            if current is None:
                return False
        return True

	#调用自：data.gaz.trie.enumerateMatch(word_list, self.space)
    def enumerateMatch(self, word, space="_", backward=False):
        matched = []
        ## while len(word) > 1 does not keep character itself, while word keed character itself
        while len(word) > 1:
            if self.search(word):  #返回在self.children字典里能不能找到key为word的value
                matched.append(space.join(word[:]))  #返回true则word这个list的所有词都在gaz_file里有匹配
            del word[-1]    #没有全部匹配到就去掉最后一个
        return matched

