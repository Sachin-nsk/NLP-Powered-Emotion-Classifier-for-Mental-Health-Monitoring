            'img',
            '''
            elements => elements.flatMap(el => {
                let urls = [];
                if(el.src) urls.push(el.src);
                if(el.srcset) {
                    urls = urls.concat(el.srcset.split(',').map(s => s.trim().split(' ')[0]));
                }
                return urls.filter(src => src.includes('instagram.fna.fbcdn.net') || src.includes('cdninstagram.com') || src.includes('scontent'));
            })
            '''
        )

        media_videos = page.eval_on_selector_all(
            'video',
            '''
            elements => elements
            .map(el => el.src)
            .filter(src => src && (src.includes('instagram.fna.fbcdn.net') || src.includes('cdninstagram.com') || src.includes('scontent')))
            '''
        )

        media_urls = list(set(media_imgs + media_videos))