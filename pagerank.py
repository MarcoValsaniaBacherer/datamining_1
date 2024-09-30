#!/usr/bin/python3

'''
This file calculates pagerank vectors for small-scale webgraphs.
See the README.md for example usage.
'''

import math
import torch
import gzip
import csv

import logging


class WebGraph():

    def __init__(self, filename, max_nnz=None, filter_ratio=None):
        '''
        Initializes the WebGraph from a file.
        The file should be a gzipped csv file.
        Each line contains two entries: the source and target corresponding to a single web link.
        This code assumes that the file is sorted on the source column.
        '''

        self.url_dict = {}
        indices = []

        from collections import defaultdict
        target_counts = defaultdict(lambda: 0)

        # loop through filename to extract the indices
        logging.debug('computing indices')
        with gzip.open(filename,newline='',mode='rt') as f:
            for i,row in enumerate(csv.DictReader(f)):
                if max_nnz is not None and i>max_nnz:
                    break
                import re
                regex = re.compile(r'.*((/$)|(/.*/)).*')
                if regex.match(row['source']) or regex.match(row['target']):
                    continue
                source = self._url_to_index(row['source'])
                target = self._url_to_index(row['target'])
                target_counts[target] += 1
                indices.append([source,target])

        # remove urls with too many in-links
        if filter_ratio is not None:
            new_indices = []
            for source,target in indices:
                if target_counts[target] < filter_ratio*len(self.url_dict):
                    new_indices.append([source,target])
            indices = new_indices

        # compute the values that correspond to the indices variable
        logging.debug('computing values')
        values = []
        last_source = indices[0][0]
        last_i = 0
        for i,(source,target) in enumerate(indices+[(None,None)]):
            if source==last_source:
                pass
            else:
                total_links = i-last_i
                values.extend([1/total_links]*total_links)
                last_source = source
                last_i = i

        # generate the sparse matrix
        i = torch.LongTensor(indices).t()
        v = torch.FloatTensor(values)
        n = len(self.url_dict)
        self.P = torch.sparse_coo_tensor(i, v, torch.Size([n,n]))
        self.index_dict = {v: k for k, v in self.url_dict.items()}
    

    def _url_to_index(self, url):
        '''
        given a url, returns the row/col index into the self.P matrix
        '''
        if url not in self.url_dict:
            self.url_dict[url] = len(self.url_dict)
        return self.url_dict[url]


    def _index_to_url(self, index):
        '''
        given a row/col index into the self.P matrix, returns the corresponding url
        '''
        return self.index_dict[index]


    def make_personalization_vector(self, query=None):
        '''
        If query is None, returns the vector of 1s.
        If query contains a string,
        then each url satisfying the query has the vector entry set to 1;
        all other entries are set to 0.
        '''
        n = self.P.shape[0]

        if query is None:
            v = torch.ones(n)

        else:
            v = torch.zeros(n)

            for i in range(n):
                 url = self._index_to_url(i)
                 if url_satisfies_query(url, query):
                     v[i] = 1

        v_sum = torch.sum(v)
        assert(v_sum>0)
        v /= v_sum

        return v

    '''
    python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query='corona'

INFO:root:rank=0 pagerank=1.6719e-01 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
INFO:root:rank=1 pagerank=1.6719e-01 url=www.lawfareblog.com/britains-coronavirus-response
INFO:root:rank=2 pagerank=1.6076e-01 url=www.lawfareblog.com/brexit-not-immune-coronavirus
INFO:root:rank=3 pagerank=1.6076e-01 url=www.lawfareblog.com/rational-security-my-corona-edition
INFO:root:rank=4 pagerank=1.6071e-01 url=www.lawfareblog.com/lawfare-podcast-united-nations-and-coronavirus-crisis
INFO:root:rank=5 pagerank=1.5632e-01 url=www.lawfareblog.com/china-responds-coronavirus-iron-grip-information-flow
INFO:root:rank=6 pagerank=1.5612e-01 url=www.lawfareblog.com/paper-hearing-experts-debate-digital-contact-tracing-and-coronavirus-privacy-concerns
INFO:root:rank=7 pagerank=1.5431e-01 url=www.lawfareblog.com/israeli-emergency-regulations-location-tracking-coronavirus-carriers
INFO:root:rank=8 pagerank=1.5401e-01 url=www.lawfareblog.com/congressional-homeland-security-committees-seek-ways-support-state-federal-responses-coronavirus
INFO:root:rank=9 pagerank=1.5317e-01 url=www.lawfareblog.com/house-oversight-committee-questions-government-response-coronavirus

python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query='corona' --search_query='-corona'
INFO:root:rank=0 pagerank=4.5760e-02 url=www.lawfareblog.com/chinatalk-how-party-takes-its-propaganda-global
INFO:root:rank=1 pagerank=2.7895e-02 url=www.lawfareblog.com/trump-cant-reopen-country-over-state-objections
INFO:root:rank=2 pagerank=2.6851e-02 url=www.lawfareblog.com/fault-lines-foreign-policy-quarantined
INFO:root:rank=3 pagerank=1.9256e-02 url=www.lawfareblog.com/chinatalk-dispatches-shanghai-beijing-and-hong-kong
INFO:root:rank=4 pagerank=1.7480e-02 url=www.lawfareblog.com/federal-courts-begin-adapt-covid-19
INFO:root:rank=5 pagerank=1.6971e-02 url=www.lawfareblog.com/lawfare-podcast-stephen-holmes-liberalism-21st-century
INFO:root:rank=6 pagerank=1.6971e-02 url=www.lawfareblog.com/fault-lines-combating-extremism-farah-pandith
INFO:root:rank=7 pagerank=1.6971e-02 url=www.lawfareblog.com/twitter-using-health-emergency-settle-political-scores
INFO:root:rank=8 pagerank=1.6477e-02 url=www.lawfareblog.com/limits-world-health-organization
INFO:root:rank=9 pagerank=1.5815e-02 url=www.lawfareblog.com/senators-urge-cyber-leaders-prevent-attacks-healthcare-sector
    '''

    def power_method(self, v=None, x0=None, alpha=0.85, max_iterations=1000, epsilon=1e-6):
        '''
        This function implements the power method for computing the pagerank.

        The self.P variable stores the $P$ matrix.
        You will have to compute the $a$ vector and implement Equation 5.1 from "Deeper Inside Pagerank."
        '''
        with torch.no_grad():
            n = self.P.shape[0]

            # create variables if none given
            if v is None:
                v = torch.Tensor([1/n]*n)
                v = torch.unsqueeze(v,1)
            v /= torch.norm(v)

            if x0 is None:
                x0 = torch.Tensor([1/(math.sqrt(n))]*n)
                x0 = torch.unsqueeze(x0,1)
            x0 /= torch.norm(x0)

            # main loop
            xprev = x0
            x = xprev.detach().clone()
            for i in range(max_iterations):
                xprev = x.detach().clone()

                # compute the new x vector using Eq (5.1)
                a = torch.ones([n,1])
                p1 = (alpha * (x.t() @ a) + (1 - alpha)) * v.t()

                # compute the new x vector using Eq (5.1)
                x = torch.sparse.addmm(p1.t(), self.P.t(), x, alpha=alpha)

                x = x / torch.norm(x)

                # output debug information
                residual = torch.norm(x-xprev)
                logging.debug(f'i={i} residual={residual}')

                # early stop when sufficient accuracy reached
                if residual < epsilon:
                    break

            #x = x0.squeeze()
            return x.squeeze()

    '''
python3 pagerank.py --data=data/small.csv.gz --verbose
DEBUG:root:computing indices
DEBUG:root:computing values
DEBUG:root:i=0 residual=0.10726700723171234
DEBUG:root:i=1 residual=0.021190769970417023
DEBUG:root:i=2 residual=0.004620412830263376
DEBUG:root:i=3 residual=0.0006995540461502969
DEBUG:root:i=4 residual=0.0001600638497620821
DEBUG:root:i=5 residual=2.3819309717509896e-05
DEBUG:root:i=6 residual=5.279218839859823e-06
DEBUG:root:i=7 residual=7.991246775418404e-07
INFO:root:rank=0 pagerank=4.9268e-01 url=4
INFO:root:rank=1 pagerank=4.3753e-01 url=6
INFO:root:rank=2 pagerank=4.1232e-01 url=5
INFO:root:rank=3 pagerank=3.9048e-01 url=2
INFO:root:rank=4 pagerank=3.5622e-01 url=3
INFO:root:rank=5 pagerank=3.4126e-01 url=1

python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='corona'
INFO:root:rank=0 pagerank=7.6612e-03 url=www.lawfareblog.com/lawfare-podcast-united-nations-and-coronavirus-crisis
INFO:root:rank=1 pagerank=7.6252e-03 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response
INFO:root:rank=2 pagerank=7.5751e-03 url=www.lawfareblog.com/britains-coronavirus-response
INFO:root:rank=3 pagerank=7.5713e-03 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
INFO:root:rank=4 pagerank=7.5664e-03 url=www.lawfareblog.com/israeli-emergency-regulations-location-tracking-coronavirus-carriers
INFO:root:rank=5 pagerank=7.5645e-03 url=www.lawfareblog.com/why-congress-conducting-business-usual-face-coronavirus
INFO:root:rank=6 pagerank=7.5597e-03 url=www.lawfareblog.com/congressional-homeland-security-committees-seek-ways-support-state-federal-responses-coronavirus
INFO:root:rank=7 pagerank=7.5551e-03 url=www.lawfareblog.com/paper-hearing-experts-debate-digital-contact-tracing-and-coronavirus-privacy-concerns
INFO:root:rank=8 pagerank=7.5488e-03 url=www.lawfareblog.com/house-subcommittee-voices-concerns-over-us-management-coronavirus
INFO:root:rank=9 pagerank=7.5474e-03 url=www.lawfareblog.com/lawfare-podcast-bonus-edition-steve-vladeck-emergency-powers-and-coronavirus

python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='trump'
INFO:root:rank=0 pagerank=8.8872e-03 url=www.lawfareblog.com/document-trump-revokes-obama-executive-order-counterterrorism-strike-casualty-reporting
INFO:root:rank=1 pagerank=8.7532e-03 url=www.lawfareblog.com/trump-administrations-worrying-new-policy-israeli-settlements
INFO:root:rank=2 pagerank=8.6518e-03 url=www.lawfareblog.com/dc-circuit-overrules-district-courts-due-process-ruling-qasim-v-trump
INFO:root:rank=3 pagerank=8.5672e-03 url=www.lawfareblog.com/how-trumps-approach-middle-east-ignores-past-future-and-human-condition
INFO:root:rank=4 pagerank=8.5591e-03 url=www.lawfareblog.com/donald-trump-and-politically-weaponized-executive-branch
INFO:root:rank=5 pagerank=8.5510e-03 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
INFO:root:rank=6 pagerank=8.5483e-03 url=www.lawfareblog.com/why-trump-cant-buy-greenland
INFO:root:rank=7 pagerank=8.4630e-03 url=www.lawfareblog.com/oral-argument-summary-qassim-v-trump
INFO:root:rank=8 pagerank=8.3594e-03 url=www.lawfareblog.com/second-circuit-rules-mazars-must-hand-over-trump-tax-returns-new-york-prosecutors
INFO:root:rank=9 pagerank=8.3579e-03 url=www.lawfareblog.com/dc-circuit-court-denies-trump-rehearing-mazars-case
    
python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='iran'
INFO:root:rank=0 pagerank=8.5985e-03 url=www.lawfareblog.com/how-us-iran-tensions-could-disrupt-iraqs-fragile-peace
INFO:root:rank=1 pagerank=8.5533e-03 url=www.lawfareblog.com/praise-presidents-iran-tweets
INFO:root:rank=2 pagerank=8.1114e-03 url=www.lawfareblog.com/cyber-command-operational-update-clarifying-june-2019-iran-operation
INFO:root:rank=3 pagerank=7.8355e-03 url=www.lawfareblog.com/aborted-iran-strike-fine-line-between-necessity-and-revenge
INFO:root:rank=4 pagerank=7.8176e-03 url=www.lawfareblog.com/parsing-state-departments-letter-use-force-against-iran
INFO:root:rank=5 pagerank=7.8123e-03 url=www.lawfareblog.com/announcing-united-states-and-use-force-against-iran-new-lawfare-e-book
INFO:root:rank=6 pagerank=7.8103e-03 url=www.lawfareblog.com/iranian-hostage-crisis-and-its-effect-american-politics
INFO:root:rank=7 pagerank=7.7858e-03 url=www.lawfareblog.com/us-names-iranian-revolutionary-guard-terrorist-organization-and-sanctions-international-criminal
INFO:root:rank=8 pagerank=7.7100e-03 url=www.lawfareblog.com/iran-shoots-down-us-drone-domestic-and-international-legal-implications
INFO:root:rank=9 pagerank=7.7036e-03 url=www.lawfareblog.com/israel-iran-syria-clash-and-law-use-force    

python3 pagerank.py --data=data/lawfareblog.csv.gz
INFO:root:rank=0 pagerank=5.9785e-02 url=www.lawfareblog.com/support-lawfare
INFO:root:rank=1 pagerank=5.9785e-02 url=www.lawfareblog.com/lawfare-job-board
INFO:root:rank=2 pagerank=5.9785e-02 url=www.lawfareblog.com/documents-related-mueller-investigation
INFO:root:rank=3 pagerank=5.9785e-02 url=www.lawfareblog.com/litigation-documents-resources-related-travel-ban
INFO:root:rank=4 pagerank=5.9785e-02 url=www.lawfareblog.com/subscribe-lawfare
INFO:root:rank=5 pagerank=5.9785e-02 url=www.lawfareblog.com/topics
INFO:root:rank=6 pagerank=5.9785e-02 url=www.lawfareblog.com/about-lawfare-brief-history-term-and-site
INFO:root:rank=7 pagerank=5.9785e-02 url=www.lawfareblog.com/our-comments-policy
INFO:root:rank=8 pagerank=5.9785e-02 url=www.lawfareblog.com/upcoming-events
INFO:root:rank=9 pagerank=5.9785e-02 url=www.lawfareblog.com/masthead

python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2
INFO:root:rank=0 pagerank=2.1205e-02 url=www.lawfareblog.com/todays-headlines-and-commentary-1964
INFO:root:rank=1 pagerank=2.1205e-02 url=www.lawfareblog.com/todays-headlines-and-commentary-1963
INFO:root:rank=2 pagerank=2.1042e-02 url=www.lawfareblog.com/todays-headlines-and-commentary-1962
INFO:root:rank=3 pagerank=1.6320e-02 url=www.lawfareblog.com/todays-headlines-and-commentary-1965
INFO:root:rank=4 pagerank=1.5234e-02 url=www.lawfareblog.com/todays-headlines-and-commentary-1966
INFO:root:rank=5 pagerank=1.5028e-02 url=www.lawfareblog.com/cyberlaw-podcast-sandworm-and-grus-global-intifada
INFO:root:rank=6 pagerank=1.5027e-02 url=www.lawfareblog.com/cyberlaw-podcast-plumbing-depths-artificial-stupidity
INFO:root:rank=7 pagerank=1.4428e-02 url=www.lawfareblog.com/lawfare-podcast-week-was-impeachment
INFO:root:rank=8 pagerank=1.4425e-02 url=www.lawfareblog.com/lawfare-podcast-ben-nimmo-whack-mole-game-disinformation
INFO:root:rank=9 pagerank=1.3983e-02 url=www.lawfareblog.com/cyberlaw-podcast-mistrusting-google

python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose 
DEBUG:root:computing indices
DEBUG:root:computing values
DEBUG:root:i=0 residual=0.18195588886737823
DEBUG:root:i=1 residual=0.00048166385386139154
DEBUG:root:i=2 residual=3.6460803585214308e-06
DEBUG:root:i=3 residual=8.956160399975488e-07
INFO:root:rank=0 pagerank=5.9785e-02 url=www.lawfareblog.com/support-lawfare
INFO:root:rank=1 pagerank=5.9785e-02 url=www.lawfareblog.com/lawfare-job-board
INFO:root:rank=2 pagerank=5.9785e-02 url=www.lawfareblog.com/documents-related-mueller-investigation
INFO:root:rank=3 pagerank=5.9785e-02 url=www.lawfareblog.com/litigation-documents-resources-related-travel-ban
INFO:root:rank=4 pagerank=5.9785e-02 url=www.lawfareblog.com/subscribe-lawfare
INFO:root:rank=5 pagerank=5.9785e-02 url=www.lawfareblog.com/topics
INFO:root:rank=6 pagerank=5.9785e-02 url=www.lawfareblog.com/about-lawfare-brief-history-term-and-site
INFO:root:rank=7 pagerank=5.9785e-02 url=www.lawfareblog.com/our-comments-policy
INFO:root:rank=8 pagerank=5.9785e-02 url=www.lawfareblog.com/upcoming-events
INFO:root:rank=9 pagerank=5.9785e-02 url=www.lawfareblog.com/masthead

python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --alpha=0.99999
DEBUG:root:computing indices
DEBUG:root:computing values
DEBUG:root:i=0 residual=0.182169109582901
DEBUG:root:i=1 residual=0.0005025434657000005
DEBUG:root:i=2 residual=4.187422746326774e-06
DEBUG:root:i=3 residual=3.725290298461914e-09
INFO:root:rank=0 pagerank=5.9852e-02 url=www.lawfareblog.com/snowden-revelations
INFO:root:rank=1 pagerank=5.9852e-02 url=www.lawfareblog.com/lawfare-job-board
INFO:root:rank=2 pagerank=5.9852e-02 url=www.lawfareblog.com/masthead
INFO:root:rank=3 pagerank=5.9852e-02 url=www.lawfareblog.com/litigation-documents-resources-related-travel-ban
INFO:root:rank=4 pagerank=5.9852e-02 url=www.lawfareblog.com/subscribe-lawfare
INFO:root:rank=5 pagerank=5.9852e-02 url=www.lawfareblog.com/litigation-documents-related-appointment-matthew-whitaker-acting-attorney-general
INFO:root:rank=6 pagerank=5.9852e-02 url=www.lawfareblog.com/documents-related-mueller-investigation
INFO:root:rank=7 pagerank=5.9852e-02 url=www.lawfareblog.com/our-comments-policy
INFO:root:rank=8 pagerank=5.9852e-02 url=www.lawfareblog.com/upcoming-events
INFO:root:rank=9 pagerank=5.9852e-02 url=www.lawfareblog.com/topics

python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --filter_ratio=0.2
DEBUG:root:computing indices
DEBUG:root:computing values
DEBUG:root:i=0 residual=0.04590300843119621
DEBUG:root:i=1 residual=0.00024055552785284817
DEBUG:root:i=2 residual=1.606495288797305e-06
DEBUG:root:i=3 residual=1.4096775657890248e-07
INFO:root:rank=0 pagerank=2.1205e-02 url=www.lawfareblog.com/todays-headlines-and-commentary-1964
INFO:root:rank=1 pagerank=2.1205e-02 url=www.lawfareblog.com/todays-headlines-and-commentary-1963
INFO:root:rank=2 pagerank=2.1042e-02 url=www.lawfareblog.com/todays-headlines-and-commentary-1962
INFO:root:rank=3 pagerank=1.6320e-02 url=www.lawfareblog.com/todays-headlines-and-commentary-1965
INFO:root:rank=4 pagerank=1.5234e-02 url=www.lawfareblog.com/todays-headlines-and-commentary-1966
INFO:root:rank=5 pagerank=1.5028e-02 url=www.lawfareblog.com/cyberlaw-podcast-sandworm-and-grus-global-intifada
INFO:root:rank=6 pagerank=1.5027e-02 url=www.lawfareblog.com/cyberlaw-podcast-plumbing-depths-artificial-stupidity
INFO:root:rank=7 pagerank=1.4428e-02 url=www.lawfareblog.com/lawfare-podcast-week-was-impeachment
INFO:root:rank=8 pagerank=1.4425e-02 url=www.lawfareblog.com/lawfare-podcast-ben-nimmo-whack-mole-game-disinformation
INFO:root:rank=9 pagerank=1.3983e-02 url=www.lawfareblog.com/cyberlaw-podcast-mistrusting-google

python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --filter_ratio=0.2 --alpha=0.99999
DEBUG:root:computing indices
DEBUG:root:computing values
DEBUG:root:i=0 residual=0.04596497491002083
DEBUG:root:i=1 residual=0.0002408745203865692
DEBUG:root:i=2 residual=1.7411409771739272e-06
DEBUG:root:i=3 residual=2.99474862686111e-08
INFO:root:rank=0 pagerank=2.1223e-02 url=www.lawfareblog.com/todays-headlines-and-commentary-1964
INFO:root:rank=1 pagerank=2.1223e-02 url=www.lawfareblog.com/todays-headlines-and-commentary-1963
INFO:root:rank=2 pagerank=2.1060e-02 url=www.lawfareblog.com/todays-headlines-and-commentary-1962
INFO:root:rank=3 pagerank=1.6331e-02 url=www.lawfareblog.com/todays-headlines-and-commentary-1965
INFO:root:rank=4 pagerank=1.5244e-02 url=www.lawfareblog.com/todays-headlines-and-commentary-1966
INFO:root:rank=5 pagerank=1.5038e-02 url=www.lawfareblog.com/cyberlaw-podcast-sandworm-and-grus-global-intifada
INFO:root:rank=6 pagerank=1.5036e-02 url=www.lawfareblog.com/cyberlaw-podcast-plumbing-depths-artificial-stupidity
INFO:root:rank=7 pagerank=1.4437e-02 url=www.lawfareblog.com/lawfare-podcast-week-was-impeachment
INFO:root:rank=8 pagerank=1.4434e-02 url=www.lawfareblog.com/lawfare-podcast-ben-nimmo-whack-mole-game-disinformation
INFO:root:rank=9 pagerank=1.3991e-02 url=www.lawfareblog.com/cyberlaw-podcast-mistrusting-google

    '''
    def search(self, pi, query='', max_results=10):
        '''
        Logs all urls that match the query.
        Results are displayed in sorted order according to the pagerank vector pi.
        '''
        n = self.P.shape[0]
        vals,indices = torch.topk(pi,n)

        matches = 0
        for i in range(n):
            if matches >= max_results:
                break
            index = indices[i].item()
            url = self._index_to_url(index)
            pagerank = vals[i].item()
            if url_satisfies_query(url,query):
                logging.info(f'rank={matches} pagerank={pagerank:0.4e} url={url}')
                matches += 1


def url_satisfies_query(url, query):
    '''
    This functions supports a moderately sophisticated syntax for searching urls for a query string.
    The function returns True if any word in the query string is present in the url.
    But, if a word is preceded by the negation sign `-`,
    then the function returns False if that word is present in the url,
    even if it would otherwise return True.

    >>> url_satisfies_query('www.lawfareblog.com/covid-19-speech', 'covid')
    True
    >>> url_satisfies_query('www.lawfareblog.com/covid-19-speech', 'coronavirus covid')
    True
    >>> url_satisfies_query('www.lawfareblog.com/covid-19-speech', 'coronavirus')
    False
    >>> url_satisfies_query('www.lawfareblog.com/covid-19-speech', 'covid -speech')
    False
    >>> url_satisfies_query('www.lawfareblog.com/covid-19-speech', 'covid -corona')
    True
    >>> url_satisfies_query('www.lawfareblog.com/covid-19-speech', '-speech')
    False
    >>> url_satisfies_query('www.lawfareblog.com/covid-19-speech', '-corona')
    True
    >>> url_satisfies_query('www.lawfareblog.com/covid-19-speech', '')
    True
    '''
    satisfies = False
    terms = query.split()

    num_terms=0
    for term in terms:
        if term[0] != '-':
            num_terms+=1
            if term in url:
                satisfies = True
    if num_terms==0:
        satisfies=True

    for term in terms:
        if term[0] == '-':
            if term[1:] in url:
                return False
    return satisfies


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--personalization_vector_query')
    parser.add_argument('--search_query', default='')
    parser.add_argument('--filter_ratio', type=float, default=None)
    parser.add_argument('--alpha', type=float, default=0.85)
    parser.add_argument('--max_iterations', type=int, default=1000)
    parser.add_argument('--epsilon', type=float, default=1e-6)
    parser.add_argument('--max_results', type=int, default=10)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    g = WebGraph(args.data, filter_ratio=args.filter_ratio)
    v = g.make_personalization_vector(args.personalization_vector_query)
    pi = g.power_method(v, alpha=args.alpha, max_iterations=args.max_iterations, epsilon=args.epsilon)
    g.search(pi, query=args.search_query, max_results=args.max_results)