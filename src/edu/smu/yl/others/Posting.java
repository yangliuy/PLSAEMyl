package edu.smu.yl.others;

class Posting
{
	int docID; // the doc where the word occur
    int pos; // the position of the word in this document

    public Posting(int id, int position)
    {
    	this.docID = id;
        this.pos = position;
    }
}
