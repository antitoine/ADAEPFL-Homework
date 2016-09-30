# How to sync this private repo with the public repo

Source : [stackoverflow.com/a/30352360](http://stackoverflow.com/a/30352360)

## Clone the private repo so you can work on it

```
git clone https://github.com/yourname/private-repo.git
cd private-repo
make some changes
git commit
git push origin master
```

## To pull new hotness from the public repo

```
cd private-repo
git remote add public https://github.com/exampleuser/public-repo.git
git pull public master # Creates a merge commit
git push origin master
```

Awesome, your private repo now has the latest code from the public repo plus your changes.

## To create a pull request private repo -> public repo

The only way to create a pull request is to have push access to the public repo. This is because you need to push to a branch there (here's why).

```
git clone https://github.com/exampleuser/public-repo.git
cd public-repo
git remote add private_repo_yourname https://github.com/yourname/private-repo.git
git checkout -b pull_request_yourname
git pull private_repo_yourname master
git push origin pull_request_yourname
```

Now simply create a pull request via the Github UI.

Once project owners review your pull request, they can merge it.

Of course the whole process can be repeated (just leave out the steps where you add remotes). 
