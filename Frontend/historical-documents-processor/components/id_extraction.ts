export function extractid(url: string): string | null {
	if(!url.match(/^https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_,\+.~#?&//=]*)$/)) {
		return null;
	}
	const id = url.split("/folders/")[1]?.split("?")[0] || null;
	return id;
}