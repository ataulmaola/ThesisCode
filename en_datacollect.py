import os
import requests
import random
import time
from bs4 import BeautifulSoup # pip install bs4

links_with_text = []

def get_link(iters):
	url='https://www.thedailystar.net/letters?page='
	link='https://www.thedailystar.net'
	for i in range(iters,iters+1):	#change 1 before run the scripts
		url=url+str(i)
		page = requests.get(url)
		content=page.content
		soup = BeautifulSoup(page.content, 'html.parser')
		soup=soup.find_all('h4')
		for a in soup:
		    hr=a.find('a', href=True,recursive=False) 
		    if hr.text: 
		        links_with_text.append(link+hr['href'])
		time.sleep(3)


def get_data(ok):
	path=os.getcwd()
	it=ok
	for item in links_with_text:
		page = requests.get(item)
		content=page.content
		soup = BeautifulSoup(page.content, 'html.parser')
		h1=soup.find('h1')
		#print(h1.text)
		mydivs = soup.find("div", {"class": "field-body view-mode-full"})
		pp=mydivs.find_all('p')
		text=''
		for i in range(len(pp)-1):
			text=text+pp[i].text+"\n"
		title_file = open(path+"/en_data/title/title_{}.txt".format(it), "w")
		title_file.write(h1.text)
		body_file=open(path+"/en_data/body/body_{}.txt".format(it), "w")
		body_file.write(text)
		title_file.close()
		body_file.close()
		time.sleep(3)
		it=it+1
print("take two inputs:\n")
in1=int(input())
in2=int(input())
get_link(in1)
# print(len(links_with_text))
# for i in links_with_text:
# 	print(i)
get_data(in2)