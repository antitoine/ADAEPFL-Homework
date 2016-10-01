# How to

## Update from origin public repository ADAEPFL/Homework

### Add the remote repository

Check that you haven't the remote upstream repository already added :
`git remote -v`

If you see something like this, you can update directly with the `upstream` remote : 
```
origin	git@github.com:antitoine/ADAEPFL-Homework.git (fetch)
origin	git@github.com:antitoine/ADAEPFL-Homework.git (push)
upstream	git@github.com:ADAEPFL/Homework.git (fetch)
upstream	git@github.com:ADAEPFL/Homework.git (push)
```

If you have only `origin`, you need to add the `upstream` like this :
`git remote add upstream git@github.com:ADAEPFL/Homework.git`

(if you prefer the https link, replace the end by that `https://github.com/ADAEPFL/Homework.git`)

Check that all is find with `git remote -v`, then you can update from the upstream repository and the origin easily (sse next step).

(There is a github note about that : [help.github.com/articles/configuring-a-remote-for-a-fork](https://help.github.com/articles/configuring-a-remote-for-a-fork/))

### Update

Just execute this after adding the remote repository (only once, see above) :
`git fetch upstream`
`git checkout master`
`git merge upstream/master`

Fix conflit if there are any, then push all on `origin` :
`git push origin master` 

(There is a github note about that : [help.github.com/articles/syncing-a-fork](https://help.github.com/articles/syncing-a-fork/))

## Before commit a notebook

Clean your notebook output to avoid extra data in the commit like the number of the ouput or the result. This is temporary because at every compilation that change.

To do so :
 - Open the notebook (like `file.ipynb`) open in Jupyter 
 - Go to the menu `Cell`
 - Then, `Current outputs`
 - And click on `Clear`

